"""Glue the four per-fold YOLO score files (40 rows each) into one 160-row file per image size.

The output has the same columns as the scores of run_experiment.py (path, label, score) and a file name
ending in _seed<N>, so eval_folds.py can read it together with the PatchCore and PaDiM scores unchanged.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores-dir", default="scores_yolo", help="folder with fold<k>_all_<size>_fold<k>.csv files")
    p.add_argument("--folds", default="splits/screw_folds.csv")
    p.add_argument("--out-dir", default="scores", help="where the merged files go (the folder eval_folds.py reads)")
    p.add_argument("--sizes", type=int, nargs="+", default=[640, 1024])
    p.add_argument("--seed", type=int, default=42, help="training seed of the YOLO runs (goes into the file name)")
    p.add_argument("--tag", default="screw_yolov8s", help="method name used in the output file name")
    return p.parse_args()


def main():
    args = parse_args()
    scores_dir, out_dir = Path(args.scores_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(args.folds)

    for size in args.sizes:
        parts = []
        for k in range(4):
            f = scores_dir / f"fold{k}_all_{size}_fold{k}.csv"
            if not f.exists():
                raise SystemExit(f"Missing file: {f}")
            part = pd.read_csv(f)
            expected = set(folds.loc[folds["fold"] == k, "path"])
            if set(part["path"]) != expected:
                raise SystemExit(f"{f.name}: images do not match fold {k} of {args.folds}")
            parts.append(part)

        merged = pd.concat(parts, ignore_index=True)
        if len(merged) != len(folds) or merged["path"].duplicated().any():
            raise SystemExit(f"imgsz {size}: expected {len(folds)} unique images, got {len(merged)}")
        check = merged.merge(folds[["path", "label"]], on="path", suffixes=("", "_folds"), validate="one_to_one")
        if (check["label"] != check["label_folds"]).any():
            raise SystemExit(f"imgsz {size}: labels differ from {args.folds}")

        out = out_dir / f"{args.tag}_img{size}_pool_seed{args.seed}.csv"
        merged.sort_values("path").to_csv(out, index=False)
        print(f"imgsz {size}: {len(merged)} rows ({int(merged['label'].sum())} defective) -> {out}")


if __name__ == "__main__":
    main()