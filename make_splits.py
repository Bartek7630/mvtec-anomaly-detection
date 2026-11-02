"""Assign every MVTec AD test image to one of K stratified folds and save the assignment to a CSV file.

The folds are created once and reused by every experiment (YOLOv8 and anomaly detection methods),
so that all methods are evaluated on exactly the same images.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="path to one MVTec AD category folder, e.g. data/mvtec/screw")
    p.add_argument("--out", default="splits/screw_folds.csv", help="output CSV file")
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def list_test_images(root: Path):
    """Return paths (relative to root, with '/' separators) and the defect type (folder name) of every test image."""
    test_dir = root / "test"
    if not test_dir.is_dir():
        raise SystemExit(f"Folder not found: {test_dir}")
    paths, defect_types = [], []
    for class_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        for image in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            paths.append(image.relative_to(root).as_posix())
            defect_types.append(class_dir.name)
    return paths, defect_types


def main():
    args = parse_args()
    root = Path(args.root)
    paths, defect_types = list_test_images(root)
    print(f"Found {len(paths)} test images: {dict(Counter(defect_types))}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_of = np.full(len(paths), -1, dtype=int)
    for fold, (_, test_index) in enumerate(skf.split(paths, defect_types)):
        fold_of[test_index] = fold
    assert (fold_of >= 0).all(), "some images were not assigned to a fold"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "defect_type", "label", "fold"])
        for path, defect_type, fold in zip(paths, defect_types, fold_of):
            writer.writerow([path, defect_type, 0 if defect_type == "good" else 1, int(fold)])

    print(f"Saved {out}\n")
    types = sorted(set(defect_types))
    print("fold | images | " + " | ".join(types))
    for fold in range(args.n_folds):
        counts = Counter(t for t, f_ in zip(defect_types, fold_of) if f_ == fold)
        print(f"{fold:>4} | {int((fold_of == fold).sum()):>6} | " + " | ".join(str(counts[t]) for t in types))


if __name__ == "__main__":
    main()
