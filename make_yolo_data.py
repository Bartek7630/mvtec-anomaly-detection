"""Build a YOLO (Ultralytics) dataset for one fold from the MVTec AD test images and their masks.

train = test images of all other folds (the pool the supervised model may learn from)
test  = test images of the chosen fold (used only for evaluation)

Defect boxes are computed from the ground-truth masks (one box per connected white region).
Good images get an empty label file, i.e. "no defect here".
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MIN_SIDE = 4   # ignore mask regions with a side (or area) of this many pixels or less
MIN_AREA = 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="path to one MVTec AD category folder, e.g. /content/screw")
    p.add_argument("--folds", default="splits/screw_folds.csv", help="CSV from make_splits.py")
    p.add_argument("--fold", type=int, required=True, help="fold used as the test set")
    p.add_argument("--out", default=None, help="output folder (default: yolo_data/fold<k>)")
    p.add_argument("--preview", type=int, default=10, help="number of images with drawn boxes to save for a visual check")
    return p.parse_args()


def boxes_from_mask(mask_path: Path, image_shape):
    """Return YOLO lines ('0 cx cy w h', normalised) for every connected region of the mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    h_img, w_img = image_shape[:2]
    if mask.shape != (h_img, w_img):
        raise ValueError(f"Mask {mask_path.name} has shape {mask.shape}, image has {(h_img, w_img)}")
    binary = (mask > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    lines, pixel_boxes = [], []
    for i in range(1, n):  # 0 is the background
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if w <= MIN_SIDE or h <= MIN_SIDE or area <= MIN_AREA:
            continue
        cx, cy = (x + w / 2) / w_img, (y + h / 2) / h_img   # centre of the BOX (not the centroid of the region)
        lines.append(f"0 {cx:.6f} {cy:.6f} {w / w_img:.6f} {h / h_img:.6f}")
        pixel_boxes.append((x, y, w, h))
    return lines, pixel_boxes


def build_split(rows: pd.DataFrame, root: Path, out: Path, split: str, previews: list):
    images_dir, labels_dir = out / "images" / split, out / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    n_boxes, no_box = 0, []
    for rel_path, defect_type in zip(rows["path"], rows["defect_type"]):
        src = root / rel_path
        name = f"{defect_type}_{src.stem}"            # avoids clashes: good/000.png vs scratch_head/000.png
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
    out = Path(args.out or f"yolo_data/fold{args.fold}")
    folds = pd.read_csv(args.folds)
    if args.fold not in set(folds["fold"]):
        raise SystemExit(f"Fold {args.fold} not found in {args.folds}")
    test_rows = folds[folds["fold"] == args.fold]
    train_rows = folds[folds["fold"] != args.fold]
    assert set(test_rows["path"]).isdisjoint(train_rows["path"]), "train and test overlap"

    if out.exists():
        shutil.rmtree(out)    # always rebuild from scratch so that files of an old fold can never leak in
    previews = []
    summary = {}
    for split, rows in (("train", train_rows), ("test", test_rows)):
        n_boxes, no_box = build_split(rows, root, out, split, previews)
        n_defective = int((rows["label"] == 1).sum())
        summary[split] = (len(rows), n_defective, n_boxes, no_box)

    yaml = (f"path: {out.resolve().as_posix()}\n"
            f"train: images/train\n"
            f"val: images/test\n"
            f"names:\n  0: defect\n")
    (out / "data.yaml").write_text(yaml)
    save_previews(previews, out, args.preview)

    print(f"Fold {args.fold} -> {out}")
    for split, (n_img, n_def, n_box, no_box) in summary.items():
        print(f"{split:>5}: {n_img} images ({n_def} defective, {n_img - n_def} good), {n_box} boxes")
        if no_box:
            print(f"       WARNING: {len(no_box)} defective images have no box: {no_box[:5]}")
    print(f"Wrote {out / 'data.yaml'} and {args.preview} preview images in {out / 'preview'}")


if __name__ == "__main__":
    main()