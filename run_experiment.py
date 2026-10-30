"""Run PatchCore / PaDiM (anomalib) on one MVTec AD category and append the metrics to a CSV file."""

import argparse
import csv
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
from anomalib.models import Padim, Patchcore

CSV_COLUMNS = [
    "data", "category", "model", "backbone", "layers", "image_size", "coreset_ratio",
    "num_neighbors", "seed", "n_train", "n_test_images",
    "image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_AUPRO", "pixel_F1Score", "fit_s", "test_s",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="path to one MVTec AD category folder, e.g. data/mvtec/screw")
    p.add_argument("--category", default="screw", help="category name (used for labelling only)")
    p.add_argument("--model", choices=["patchcore", "padim"], default="patchcore")
    p.add_argument("--backbone", default=None,
                   help="default: wide_resnet50_2 for patchcore, resnet18 for padim")
    p.add_argument("--n-features", type=int, default=None, help="PaDiM only: number of randomly selected channels")
    p.add_argument("--image-size", type=int, default=256, help="input resolution (square)")
    p.add_argument("--coreset-ratio", type=float, default=0.01, help="PatchCore: fraction of patches kept in the memory bank")
    p.add_argument("--num-neighbors", type=int, default=9, help="PatchCore: neighbours used for the image-level score")
    p.add_argument("--seeds", type=int, nargs="+", default=[42], help="one run per seed")
    p.add_argument("--accelerator", default="auto", help="auto, cpu or gpu")
    p.add_argument("--out", default="results.csv", help="CSV file to append results to")
    p.add_argument("--results-dir", default="runs", help="folder for anomalib output and error-case figures")
    p.add_argument("--save-errors", action="store_true", help="save a figure with the worst false positive and false negative")
    p.add_argument("--no-pretrained", action="store_true", help="random backbone weights (smoke tests only)")
    p.add_argument("--n-train", type=int, default=None, help="few-shot: randomly sample this many training images per seed")
    args = p.parse_args()
    if args.n_train is not None and args.n_train < 1:
        p.error("--n-train must be >= 1")
    if args.model == "padim" and args.n_train == 1:
        p.error("PaDiM needs at least 2 training images")
    if args.backbone is None:
        args.backbone = "wide_resnet50_2" if args.model == "patchcore" else "resnet18"
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


def save_error_cases(engine, model, datamodule, out_png):
    """Save the good image with the highest score (false positive) and the defective one with the lowest (false negative)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

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
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        seed=seed,
        num_workers=0,
        train_batch_size=16,
        eval_batch_size=16,
    )

    evaluator = Evaluator(test_metrics=[
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
        AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
        F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
    ])

    model_cls = Patchcore if args.model == "patchcore" else Padim
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
    else:
        model = Padim(
            backbone=args.backbone,
            layers=["layer1", "layer2", "layer3"],
            pre_trained=not args.no_pretrained,
            n_features=args.n_features,
            pre_processor=pre_processor,
            evaluator=evaluator,
        )
    engine = Engine(accelerator=args.accelerator, default_root_dir=args.results_dir)

    t0 = time.time()
    engine.fit(model=model, datamodule=datamodule)
    t_fit = time.time() - t0

    t0 = time.time()
    metrics = engine.test(model=model, datamodule=datamodule)[0]
    t_test = time.time() - t0

    if args.save_errors:
        name = f"errors_{args.model}_{args.backbone}_{args.image_size}_n{args.n_train or 'all'}_seed{seed}.png"
        save_error_cases(engine, model, datamodule, Path(args.results_dir) / name)

    return {
        "data": "MVTecAD", "category": args.category, "model": args.model, "backbone": args.backbone,
        "layers": "layer2+layer3" if args.model == "patchcore" else (
            "layer1+layer2+layer3" + (f" d={args.n_features}" if args.n_features else "")),
        "image_size": args.image_size,
        "coreset_ratio": args.coreset_ratio if args.model == "patchcore" else None,
        "num_neighbors": args.num_neighbors if args.model == "patchcore" else None, "seed": seed,
        "n_train": len(datamodule.train_data), "n_test_images": len(datamodule.test_data),
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