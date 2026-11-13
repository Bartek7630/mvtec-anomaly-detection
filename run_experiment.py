"""Run PatchCore / PaDiM / STFPM / EfficientAD (anomalib) on one MVTec AD category and append the metrics to a CSV file."""

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path
import random
import shutil
import tempfile

import torch
from lightning.pytorch import seed_everything

from anomalib.metrics import AUROC, AUPRO, F1Score, Evaluator
from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, Stfpm, EfficientAd, WinClip


def _collect_reference_images_fixed(self, dataloader):
    images = []
    for item in dataloader.dataset:
        images.append(item.image.unsqueeze(0))
        if len(images) == self.k_shot:
            break
    return torch.cat(images)


WinClip.collect_reference_images = _collect_reference_images_fixed

CSV_COLUMNS = [
    "data", "category", "model", "backbone", "layers", "image_size", "coreset_ratio",
    "num_neighbors", "epochs", "steps", "seed", "val_split", "n_train", "n_test_images",
    "image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_AUPRO", "pixel_F1Score", "fit_s", "test_s",
]


TRAIN_BATCH = 16     # training batch size (PatchCore, PaDiM, STFPM); steps per epoch = ceil(images / TRAIN_BATCH)
EFFICIENTAD_STEPS = 70_000   # training steps in the EfficientAD paper (batch size 1)
EFFICIENTAD_VAL_RATIO = 0.1  # share of the training images used for the map-normalisation quantiles (--val-split from_train)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="path to one MVTec AD category folder, e.g. data/mvtec/screw")
    p.add_argument("--category", default="screw", help="category name")
    p.add_argument("--model", choices=["patchcore", "padim", "stfpm", "efficientad", "winclip"], default="patchcore")
    p.add_argument("--efficientad-modelsize", default="s", choices=["s", "m"], help="EfficientAD: size of teacher and student")
    p.add_argument("--k-shot", type=int, default=0, help="WinClip: number of shots to use for training (0 means no shots)")
    p.add_argument("--imagenet-dir", default="datasets/imagenette",
                   help="EfficientAD: folder with ImageNette")
    p.add_argument("--val-split", choices=["auto", "same_as_test", "from_train"], default="auto",
                   help="where the validation images come from. auto = same_as_test for PatchCore/PaDiM/STFPM (unchanged) and "
                        "from_train for EfficientAD (10%% of the good training images; see the note in the README)")
    p.add_argument("--backbone", default=None,
                   help="default: wide_resnet50_2 for patchcore, resnet18 for padim, resnet18 for stfpm")
    p.add_argument("--n-features", type=int, default=None, help="PaDiM only: number of randomly selected channels")
    p.add_argument("--image-size", type=int, default=256, help="input resolution (square)")
    p.add_argument("--coreset-ratio", type=float, default=0.01, help="PatchCore: fraction of patches kept in the memory bank")
    p.add_argument("--num-neighbors", type=int, default=9, help="PatchCore: neighbours used for the image-level score")
    p.add_argument("--seeds", type=int, nargs="+", default=[42], help="one run per seed")
    p.add_argument("--accelerator", default="auto", help="auto, cpu or gpu")
    p.add_argument("--out", default="results.csv", help="CSV file to append results to")
    p.add_argument("--results-dir", default="runs", help="folder for anomalib output and error-case figures")
    p.add_argument("--save-errors", action="store_true", help="save a figure with the worst false positive and false negative")
    p.add_argument("--save-scores", action="store_true", help="save the score of every test image to a CSV file")
    p.add_argument("--scores-dir", default="scores", help="folder for the per-image score files")
    p.add_argument("--no-pretrained", action="store_true", help="random backbone weights (smoke tests only)")
    p.add_argument("--n-train", type=int, default=None, help="few-shot: randomly sample this many training images per seed")
    p.add_argument("--epochs", type=int, default=None,
                   help="STFPM / EfficientAD: number of training epochs (default: 100 for STFPM, as in the paper; for EfficientAD 70000 steps)")
    p.add_argument("--iterations", type=int, default=None,
                   help="STFPM / EfficientAD: target number of training steps. The number of epochs is then computed as "
                        "ceil(iterations / steps_per_epoch) and overrides --epochs (STFPM: steps_per_epoch = ceil(n_images / 16), "
                        "100 epochs on 320 images = 2000 steps; EfficientAD: batch size 1, so steps_per_epoch = n_images)")
    p.add_argument("--val-every", type=int, default=None,
                   help="STFPM / EfficientAD: validate every N epochs (default: only after the last epoch, which saves time)")
    args = p.parse_args()
    trained = args.model in ("stfpm", "efficientad")
    if trained and args.no_pretrained:
        p.error(f"--no-pretrained is not supported for {args.model} (its teacher is always pretrained)")
    if args.model == "efficientad" and args.backbone is not None:
        p.error("--backbone is not used by efficientad (use --efficientad-modelsize)")
    if args.epochs is not None and args.epochs < 1:
        p.error("--epochs must be >= 1")
    if args.iterations is not None and (not trained or args.iterations < 1):
        p.error("--iterations needs --model stfpm or efficientad and a value >= 1")
    if args.model == "stfpm" and args.epochs is None and args.iterations is None:
        args.epochs = 100
    if args.model == "efficientad" and args.epochs is None and args.iterations is None:
        args.iterations = EFFICIENTAD_STEPS
    if args.epochs is None:
        args.epochs = 1
    if args.val_split == "auto":
        args.val_split = "from_train" if args.model == "efficientad" else "same_as_test"
    if args.model == "efficientad" and args.n_train is not None and args.val_split == "from_train":
        p.error("--n-train with efficientad needs --val-split same_as_test (the validation images are taken from the training images)")
    if args.model == "winclip":
        if args.k_shot == 0 and args.n_train is not None:
            p.error("--n-train for winclip needs --k-shot N (with --k-shot 0 the model sees no training images)")
        if args.k_shot > 0:
            if args.n_train is None:
                args.n_train = args.k_shot
            elif args.n_train != args.k_shot:
                p.error(f"--n-train ({args.n_train}) must match --k-shot ({args.k_shot}) for winclip")
        elif args.k_shot < 0:
            p.error("--k-shot must be >= 0")
        if args.image_size == 256:
            args.image_size = 240
    if args.n_train is not None and args.n_train < 1:
        p.error("--n-train must be >= 1")
    if args.model == "padim" and args.n_train == 1:
        p.error("PaDiM needs at least 2 training images")
    if args.backbone is None:
        args.backbone = (
            "wide_resnet50_2" if args.model == "patchcore"
            else f"pdn_{args.efficientad_modelsize}" if args.model == "efficientad"
            else "ViT-B-16-plus-240" if args.model == "winclip"
            else "resnet18"
        )
    return args


def find_defects(root: Path):
    """Return the defect image folders (everything in test/ except 'good') and the matching mask folders."""
    defects = sorted(d.name for d in (root / "test").iterdir() if d.is_dir() and d.name != "good")
    return [f"test/{d}" for d in defects], [f"ground_truth/{d}" for d in defects]


def make_train_subset(src_dir: Path, n: int, seed: int, category: str) -> Path:
    """Copy n randomly chosen training images (reproducible for a given seed) to a fresh temporary folder."""
    files = sorted(p for p in Path(src_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    chosen = random.Random(seed).sample(files, min(n, len(files)))
    out = Path(tempfile.mkdtemp(prefix=f"fewshot_{category}_n{n}_seed{seed}_"))
    for f in chosen:
        shutil.copy(f, out / f.name)
    return out


def collect_predictions(engine, model, datamodule):
    """Run the model on the test images and return one dict per image (path, true label, score, map, mask)."""
    items = []
    for b in engine.predict(model=model, datamodule=datamodule, return_predictions=True):
        for i, path in enumerate(b.image_path):
            items.append({
                "path": path,
                "score": float(b.pred_score[i]),
                "label": int(b.gt_label[i]),
                "pred": int(b.pred_label[i]),
                "amap": b.anomaly_map[i].cpu().numpy(),
                "mask": None if b.gt_mask is None else b.gt_mask[i].cpu().numpy(),
            })
    return items


def scores_file(args, seed):
    """File name that identifies the run, so that runs with different settings never overwrite each other."""
    parts = [args.category, args.model, args.backbone, f"s{args.image_size}"]
    if args.model == "patchcore":
        parts.append(f"cr{args.coreset_ratio}")
    elif args.n_features:
        parts.append(f"d{args.n_features}")
    if args.model in ("stfpm", "efficientad"):
        parts.append(f"e{args.epochs}")
    n_str = "0" if (args.model == "winclip" and args.k_shot == 0) else (args.n_train or "all")
    parts += [f"n{n_str}", f"seed{seed}"]
    return Path(args.scores_dir) / ("_".join(parts) + ".csv")


def save_scores(items, out_csv):
    """Save path (relative to the category folder, e.g. test/good/000.png), true label and score of every test image."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score"])
        for x in items:
            w.writerow(["/".join(Path(x["path"]).parts[-3:]), x["label"], round(x["score"], 6)])
    print("Saved scores:", out_csv)


def save_error_cases(items, out_png):
    """Save the good image with the highest score (false positive) and the defective one with the lowest (false negative)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    good = [x for x in items if x["label"] == 0]
    bad = [x for x in items if x["label"] == 1]
    cases = [
        ("False Positive", max(good, key=lambda x: x["score"])),
        ("False Negative", min(bad, key=lambda x: x["score"])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for row, (title, x) in enumerate(cases):
        h, w = x["amap"].shape
        img = np.asarray(Image.open(x["path"]).convert("RGB").resize((w, h)))
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{title}\n{Path(x['path']).parent.name}/{Path(x['path']).name}", fontsize=8)
        axes[row, 1].imshow(img)
        axes[row, 1].imshow(x["amap"], cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[row, 1].set_title(f"anomaly map | score {x['score']:.2f} | model says: "
                               f"{'DEFECT' if x['pred'] else 'GOOD'}", fontsize=8)
        axes[row, 2].imshow(x["mask"] if x["mask"] is not None else np.zeros((h, w)), cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("ground-truth mask" + ("" if x["label"] else " (good photo)"), fontsize=8)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print("Saved figure:", out_png)


def _run(args, seed, normal_dir):
    """Fit and test one model for one seed and return a dict with one CSV row."""
    seed_everything(seed)
    root = Path(args.root)
    abnormal_dir, mask_dir = find_defects(root)
    if args.n_train and args.model == "patchcore" and args.coreset_ratio < 1.0:
        k = int(args.coreset_ratio * args.n_train * 1024)
        print(f"Warning: with N={args.n_train} and coreset-ratio={args.coreset_ratio} the memory bank "
              f"has only {k} patches. For few-shot runs consider --coreset-ratio 1.0.")
    datamodule = Folder(
        name=args.category,
        root=root,
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir,
        normal_test_dir="test/good",
        mask_dir=mask_dir,
        val_split_mode=ValSplitMode.FROM_TRAIN if args.val_split == "from_train" else ValSplitMode.SAME_AS_TEST,
        val_split_ratio=EFFICIENTAD_VAL_RATIO if args.val_split == "from_train" else 0.5,
        seed=seed,
        num_workers=0,
        train_batch_size=1 if args.model == "efficientad" else TRAIN_BATCH,
        eval_batch_size=16,
    )

    evaluator = Evaluator(test_metrics=[
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
        AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
    ])

    model_cls = {"patchcore": Patchcore, "stfpm": Stfpm, "padim": Padim, "efficientad": EfficientAd, "winclip":WinClip}[args.model]
    if args.model == "winclip":
        pre_processor = True
    else:
        pre_processor = True if args.image_size == 256 else model_cls.configure_pre_processor(
            image_size=(args.image_size, args.image_size))

    if args.model == "patchcore":
        model = Patchcore(
            backbone=args.backbone,
            layers=("layer2", "layer3"),
            pre_trained=not args.no_pretrained,
            coreset_sampling_ratio=args.coreset_ratio,
            num_neighbors=args.num_neighbors,
            pre_processor=pre_processor,
            evaluator=evaluator,
        )
    elif args.model == "stfpm":
        model = Stfpm(
            backbone=args.backbone,
            layers=("layer1", "layer2", "layer3"),
            pre_processor=pre_processor,
            evaluator=evaluator,
        )
    elif args.model == "padim":
        model = Padim(
            backbone=args.backbone,
            layers=["layer1", "layer2", "layer3"],
            pre_trained=not args.no_pretrained,
            n_features=args.n_features,
            pre_processor=pre_processor,
            evaluator=evaluator,
        )
    elif args.model == "efficientad":
        model = EfficientAd(
            imagenet_dir=args.imagenet_dir,
            model_size={"s": "small", "m": "medium"}[args.efficientad_modelsize],
            pre_processor=pre_processor,
            evaluator=evaluator,
        )
    elif args.model == "winclip":
        ref_source = str(Path(normal_dir) if Path(normal_dir).is_absolute() else root / normal_dir) if args.k_shot > 0 else None
        model = WinClip(
            class_name=args.category,
            k_shot=args.k_shot,
            few_shot_source=ref_source,
            pre_processor=pre_processor,
            evaluator=evaluator,
        )
        if not hasattr(model.pre_processor, "test_transform"):
            object.__setattr__(model.pre_processor, "test_transform", model.pre_processor.transform)
        

    steps_per_epoch = None
    if args.model == "stfpm":
        folder = Path(normal_dir) if Path(normal_dir).is_absolute() else root / normal_dir
        n_images = sum(1 for f in Path(folder).iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        steps_per_epoch = math.ceil(n_images / TRAIN_BATCH)
    elif args.model == "efficientad":
        datamodule.setup("fit")
        n_images = len(datamodule.train_data)
        steps_per_epoch = n_images
    if steps_per_epoch:
        if args.iterations:
            args.epochs = math.ceil(args.iterations / steps_per_epoch)
        print(f"{args.model}: {n_images} training images, {steps_per_epoch} steps per epoch, {args.epochs} epochs "
              f"= {args.epochs * steps_per_epoch} steps")
        val_every = args.val_every or args.epochs
        if args.epochs % val_every != 0 and args.model == "efficientad":
            raise SystemExit(f"--val-every {val_every} does not divide the {args.epochs} epochs: the last epoch would not be validated")
        engine = Engine(accelerator=args.accelerator, default_root_dir=args.results_dir, max_epochs=args.epochs,
                        check_val_every_n_epoch=val_every)
    else:
        engine = Engine(accelerator=args.accelerator, default_root_dir=args.results_dir)

    t_fit = 0.0
    if args.model != "winclip":
        t0 = time.time()
        engine.fit(model=model, datamodule=datamodule)
        t_fit = time.time() - t0

    t0 = time.time()
    metrics = engine.test(model=model, datamodule=datamodule)[0]
    t_test = time.time() - t0

    if args.save_errors or args.save_scores:
        items = collect_predictions(engine, model, datamodule)
        if args.save_scores:
            save_scores(items, scores_file(args, seed))
        if args.save_errors:
            n_str = "0" if (args.model == "winclip" and args.k_shot == 0) else (args.n_train or "all")
            name = f"errors_{args.model}_{args.backbone}_{args.image_size}_n{n_str}_seed{seed}.png"
            save_error_cases(items, Path(args.results_dir) / name)

    return {
        "data": "MVTecAD", "category": args.category, "model": args.model, "backbone": args.backbone,
        "layers": (
            "layer2+layer3" if args.model == "patchcore"
            else "pdn" if args.model == "efficientad"
            else "clip" if args.model == "winclip"
            else ("layer1+layer2+layer3" + (f" d={args.n_features}" if args.n_features else ""))),
        "image_size": args.image_size,
        "coreset_ratio": args.coreset_ratio if args.model == "patchcore" else None,
        "num_neighbors": args.num_neighbors if args.model == "patchcore" else None,
        "epochs": args.epochs if steps_per_epoch else None,
        "steps": args.epochs * steps_per_epoch if steps_per_epoch else None, "seed": seed, "val_split": args.val_split,
        "n_train": 0 if (args.model == "winclip" and args.k_shot == 0) else len(datamodule.train_data),
        "n_test_images": len(datamodule.test_data),
        "image_AUROC": metrics.get("image_AUROC"), "image_F1Score": metrics.get("image_F1Score"),
        "pixel_AUROC": metrics.get("pixel_AUROC"), "pixel_AUPRO": metrics.get("pixel_AUPRO"),
        "pixel_F1Score": metrics.get("pixel_F1Score"),
        "fit_s": round(t_fit, 1), "test_s": round(t_test, 1),
    }


def run_one(args, seed):
    """Run one seed; in few-shot mode the temporary training subset is always deleted afterwards."""
    subset = None
    try:
        if args.n_train:
            subset = make_train_subset(Path(args.root) / "train" / "good", args.n_train, seed, args.category)
        return _run(args, seed, normal_dir=subset if subset else "train/good")
    finally:
        if subset:
            shutil.rmtree(subset, ignore_errors=True)


def _prepare_csv(path):
    """Check the CSV header: append if it matches, migrate an older header, otherwise write to a new file."""
    path = Path(path)
    if not path.exists():
        return path
    with open(path, encoding="utf-8-sig", newline="") as f:
        first = f.readline().strip()
    if first == ",".join(CSV_COLUMNS):
        return path
    old_cols = first.split(",")
    if ";" not in first and all(c in CSV_COLUMNS for c in old_cols):
        backup = path.with_name(path.stem + "_before_migration" + path.suffix)
        shutil.copy(path, backup)
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
        print(f"\n!!! {path.name}: added missing columns (backup of the old file: {backup.name}).\n")
        return path
    new_path = path.with_name(path.stem + "_new" + path.suffix)
    print(f"\n!!! {path.name} has an unknown header. Saving to {new_path.name}\n")
    return new_path


def _write_row(path, row):
    path = _prepare_csv(path)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


def append_row(path, row):
    """Append one row; if the file is locked (e.g. open in Excel) write to a timestamped backup file instead."""
    try:
        _write_row(path, row)
    except PermissionError:
        backup = f"results_backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
        _write_row(backup, row)
        print(f"\n!!! Cannot write to {path}. Saved to {backup} instead.\n")


if __name__ == "__main__":
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    print("CUDA available:", torch.cuda.is_available(),
          "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
          "| torch:", torch.__version__, "| --accelerator:", args.accelerator)
    for seed in args.seeds:
        row = run_one(args, seed)
        append_row(args.out, row)
        print("Saved:", row)