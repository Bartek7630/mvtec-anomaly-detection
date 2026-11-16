"""Honest decision threshold: set it on GOOD images from other folds, measure recall and false alarms on the held-out fold.

Why: F1 with the threshold chosen on the test images themselves is optimistic. A factory has (almost) only good
images when it sets up the system, so the realistic rule is "accept about 5% false alarms on good parts".

For every run (one file in scores/) and every fold k:
    1. threshold = 95th percentile of the scores of the GOOD test images in the other folds,
    2. on fold k: recall = share of defective images with score >= threshold,
                  false alarm rate = share of good images with score >= threshold.
Counts are pooled over the 4 folds (so the numbers are for all 160 images, each judged by a threshold that never saw it).
For comparison it also reports the optimistic F1 (best threshold chosen on all images) next to the honest F1.

Usage (repo root):
    python analysis/threshold_by_folds.py --category screw
Input : scores/<category>_*.csv (path,label,score) and splits/<category>_folds.csv (path,defect_type,label,fold)
Output: results/threshold_by_folds_<category>.csv (one row per method, mean over seeds)
        results/threshold_by_folds_<category>_per_run.csv (one row per file)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]


def f1_from_counts(tp, fp, fn):
    """F1 = 2TP / (2TP + FP + FN); 0 if there is nothing to divide."""
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def optimistic_f1(labels, scores):
    """Best F1 over all possible thresholds, chosen on the very same images (optimistic)."""
    p, r, _ = precision_recall_curve(labels, scores)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return float(f.max())


def run_one(df, level):
    """df: one run, columns label, score, fold (160 rows). Returns a dict of pooled results."""
    tp = fp = fn = tn = 0
    thresholds = []
    for k in sorted(df["fold"].unique()):
        train, test = df[df["fold"] != k], df[df["fold"] == k]
        thr = np.percentile(train.loc[train["label"] == 0, "score"], level)
        thresholds.append(thr)
        flagged = test["score"].to_numpy() >= thr
        y = test["label"].to_numpy()
        tp += int((flagged & (y == 1)).sum())
        fn += int((~flagged & (y == 1)).sum())
        fp += int((flagged & (y == 0)).sum())
        tn += int((~flagged & (y == 0)).sum())
    return dict(recall=tp / (tp + fn), false_alarm=fp / (fp + tn), precision=tp / (tp + fp) if tp + fp else np.nan,
                f1_honest=f1_from_counts(tp, fp, fn), f1_optimistic=optimistic_f1(df["label"], df["score"]),
                thr_min=min(thresholds), thr_max=max(thresholds), tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", default="screw")
    p.add_argument("--scores-dir", default=str(ROOT / "scores"))
    p.add_argument("--folds", default=None, help="default: splits/<category>_folds.csv")
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--level", type=float, default=95.0, help="percentile of good scores used as the threshold")
    args = p.parse_args()

    folds = pd.read_csv(args.folds or ROOT / "splits" / f"{args.category}_folds.csv")
    folds = folds.rename(columns={"label": "label_folds"})[["path", "label_folds", "fold"]]

    rows = []
    for f in sorted(Path(args.scores_dir).glob(f"{args.category}_*.csv")):
        d = pd.read_csv(f).merge(folds, on="path", how="left", validate="many_to_one")
        if d["fold"].isna().any() or len(d) != len(folds):
            print(f"skip {f.name}: does not match the folds file ({len(d)} rows)")
            continue
        assert (d["label"] == d["label_folds"]).all(), f.name
        method, _, seed = f.stem.rpartition("_seed")
        rows.append(dict(method=method, seed=int(seed), **run_one(d, args.level)))
    if not rows:
        raise SystemExit("no score files matched")

    per_run = pd.DataFrame(rows)
    num = ["recall", "false_alarm", "precision", "f1_honest", "f1_optimistic"]
    summary = per_run.groupby("method").agg(seeds=("seed", "count"), **{c: (c, "mean") for c in num},
                                            recall_std=("recall", "std")).reset_index()
    summary["f1_gap"] = summary["f1_optimistic"] - summary["f1_honest"]
    summary = summary.sort_values("recall", ascending=False)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_run.round(4).to_csv(out / f"threshold_by_folds_{args.category}_per_run.csv", index=False)
    summary.round(4).to_csv(out / f"threshold_by_folds_{args.category}.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
