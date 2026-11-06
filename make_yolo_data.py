"""Build a YOLO (Ultralytics) dataset for one fold from the MVTec AD test images and their masks.

train = test images of all other folds (the pool the supervised model may learn from)
test  = test images of the chosen fold (used only for evaluation)

With --n-defect N only N randomly chosen defective images of the pool (and the same number of good images, if the
pool has that many) are used for training. The same --seed always gives the same images. The test set never changes.

Defect boxes are computed from the ground-truth masks (one box per connected white region).
Good images get an empty label file, i.e. "no defect here".
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MIN_SIDE = 4
MIN_AREA = 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="path to one MVTec AD category folder, e.g. /content/screw")
    p.add_argument("--folds", default="splits/screw_folds.csv", help="CSV from make_splits.py")
    p.add_argument("--fold", type=int, required=True, help="fold used as the test set")
    p.add_argument("--out", default=None,
                   help="output folder (default: yolo_data/fold<k>, or yolo_data/fold<k>_n<N>_seed<S> with --n-defect)")
    p.add_argument("--preview", type=int, default=10, help="number of images with drawn boxes to save for a visual check")
    p.add_argument("--n-defect", type=int, default=None,
                   help="few-shot: use only this many defective training images (default: all defective images of the pool)")
    p.add_argument("--seed", type=int, default=42, help="which images --n-defect picks (same seed = same images)")
    args = p.parse_args()
    if args.n_defect is not None and args.n_defect < 1:
        p.error("--n-defect must be >= 1")
    return args


def boxes_from_mask(mask_path: Path, image_shape):
    """Return YOLO lines ('0 cx cy w h', normalised) for every connected region of the mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    h_img, w_img = image_shape[:2]
    if mask.shape != (h_img, w_img):
        raise ValueError(f"Mask {mask_path.name} has shape {mask.shape}, image has {(h_img, w_img)}")
    binary = (mask > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    lines, pixel_boxes = [], []
    for i in range(1, n):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if w <= MIN_SIDE or h <= MIN_SIDE or area <= MIN_AREA:
            continue
        cx, cy = (x + w / 2) / w_img, (y + h / 2) / h_img
        lines.append(f"0 {cx:.6f} {cy:.6f} {w / w_img:.6f} {h / h_img:.6f}")
        pixel_boxes.append((x, y, w, h))
    return lines, pixel_boxes


def choose_train_rows(pool: pd.DataFrame, n_defect, seed: int) -> pd.DataFrame:
    """Return the training rows: the whole pool, or n_defect random defective images plus as many good ones."""
    if n_defect is None:
        return pool
    rng = random.Random(seed)
    defective = pool[pool["label"] == 1].sort_values("path")
    good = pool[pool["label"] == 0].sort_values("path")
    n = min(n_defect, len(defective))
    if n < n_defect:
        print(f"WARNING: asked for {n_defect} defective images, the pool has only {len(defective)}; using all of them.")
    chosen_defective = defective.iloc[sorted(rng.sample(range(len(defective)), n))]
    n_good = min(n, len(good))
    chosen_good = good.iloc[sorted(rng.sample(range(len(good)), n_good))]
    return pd.concat([chosen_defective, chosen_good])


def build_split(rows: pd.DataFrame, root: Path, out: Path, split: str, previews: list):
    images_dir, labels_dir = out / "images" / split, out / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    n_boxes, no_box = 0, []
    for rel_path, defect_type in zip(rows["path"], rows["defect_type"]):
        src = root / rel_path
        name = f"{defect_type}_{src.stem}"
        shutil.copy2(src, images_dir / f"{name}{src.suffix}")
        label_file = labels_dir / f"{name}.txt"
        if defect_type == "good":
            label_file.write_text("")
            continue
        image = cv2.imread(str(src))
        mask_path = root / "ground_truth" / defect_type / f"{src.stem}_mask.png"
        lines, pixel_boxes = boxes_from_mask(mask_path, image.shape)
        label_file.write_text("\n".join(lines) + ("\n" if lines else ""))
        n_boxes += len(lines)
        if not lines:
            no_box.append(rel_path)
        previews.append((image, pixel_boxes, name, split))
    return n_boxes, no_box


def save_previews(previews, out: Path, n: int):
    folder = out / "preview"
    folder.mkdir(exist_ok=True)
    step = max(1, len(previews) // max(n, 1))
    for image, boxes, name, split in previews[::step][:n]:
        for x, y, w, h in boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(str(folder / f"{split}_{name}.png"), image)


def main():
    args = parse_args()
    root = Path(args.root)
    default_name = f"fold{args.fold}" if args.n_defect is None else f"fold{args.fold}_n{args.n_defect}_seed{args.seed}"
    out = Path(args.out or f"yolo_data/{default_name}")
    folds = pd.read_csv(args.folds)
    if args.fold not in set(folds["fold"]):
        raise SystemExit(f"Fold {args.fold} not found in {args.folds}")
    test_rows = folds[folds["fold"] == args.fold]
    pool = folds[folds["fold"] != args.fold]
    train_rows = choose_train_rows(pool, args.n_defect, args.seed)
    assert set(test_rows["path"]).isdisjoint(train_rows["path"]), "train and test overlap"

    if out.exists():
        shutil.rmtree(out)
    previews = []
    summary = {}
    for split, rows in (("train", train_rows), ("test", test_rows)):
        n_boxes, no_box = build_split(rows, root, out, split, previews)
        n_defective = int((rows["label"] == 1).sum())
        summary[split] = (len(rows), n_defective, n_boxes, no_box)

    yaml = (f"path: {out.resolve().as_posix()}\n"
            f"train: images/train\n"
            f"val: images/test\n"
            f"test: images/test\n"
            f"names:\n  0: defect\n")
    (out / "data.yaml").write_text(yaml)
    save_previews(previews, out, args.preview)

    print(f"Fold {args.fold} -> {out}")
    for split, (n_img, n_def, n_box, no_box) in summary.items():
        print(f"{split:>5}: {n_img} images ({n_def} defective, {n_img - n_def} good), {n_box} boxes")
        if no_box:
            print(f"       WARNING: {len(no_box)} defective images have no box: {no_box[:5]}")
    if args.n_defect is not None:
        types = Counter(train_rows.loc[train_rows["label"] == 1, "defect_type"])
        print(f"Chosen defect types (seed {args.seed}): {dict(types)}")
    print(f"Wrote {out / 'data.yaml'} and {args.preview} preview images in {out / 'preview'}")


if __name__ == "__main__":
    main()