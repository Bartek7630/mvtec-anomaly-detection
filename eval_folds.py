"""Compute image-level AUROC separately for every fold from the per-image score files.

Input:  scores/*.csv (columns path, label, score; written by run_experiment.py --save-scores)
        splits/screw_folds.csv (columns path, defect_type, label, fold; written by make_splits.py)
Output: a table with AUROC per method and fold, its mean and the standard deviation across folds.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores-dir", default="scores", help="folder with the per-image score files")
    p.add_argument("--folds", default="splits/screw_folds.csv", help="CSV file with the fold of every test image")
    p.add_argument("--out-dir", default="results", help="folder for the output tables")
    return p.parse_args()


def load_scores(scores_dir: Path) -> pd.DataFrame:
    """Read all score files and stack them into one table with an extra 'run' column (the file name)."""
    files = sorted(scores_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No score files found in {scores_dir}")
    tables = []
    for file in files:
        table = pd.read_csv(file)
        table["run"] = file.stem
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def check_merge(merged: pd.DataFrame, n_images: int) -> None:
    """Stop if the scores and the folds do not match each other."""
    if merged["fold"].isna().any():
        bad = merged.loc[merged["fold"].isna(), "path"].unique()[:5]
        raise SystemExit(f"Some images have no fold (paths do not match). Examples: {list(bad)}")
    sizes = merged.groupby("run").size()
    wrong = sizes[sizes != n_images]
    if not wrong.empty:
        raise SystemExit(f"These runs do not have {n_images} images: {wrong.to_dict()}")
    if (merged["label"] != merged["label_folds"]).any():
        raise SystemExit("True labels in the score files differ from the labels in the folds file.")


def fold_auroc(group: pd.DataFrame) -> float:
    """AUROC of one fold; NaN if the fold has only one class."""
    if group["label"].nunique() < 2:
        return float("nan")
    return roc_auc_score(group["label"], group["score"])


def main():
    args = parse_args()
    folds = pd.read_csv(args.folds).rename(columns={"label": "label_folds"})
    scores = load_scores(Path(args.scores_dir))

    merged = scores.merge(folds[["path", "label_folds", "fold"]], on="path", how="left", validate="many_to_one")
    check_merge(merged, n_images=len(folds))

    per_run = (merged.groupby(["run", "fold"]).apply(fold_auroc, include_groups=False)
               .rename("auroc").reset_index())
    per_run["method"] = per_run["run"].str.rsplit("_seed", n=1).str[0]
    per_run["seed"] = per_run["run"].str.rsplit("_seed", n=1).str[-1]

    # mean over seeds inside each fold, then mean and std across folds
    per_fold = per_run.groupby(["method", "fold"])["auroc"].mean().unstack("fold")
    per_fold.columns = [f"fold_{int(c)}" for c in per_fold.columns]
    summary = per_fold.copy()
    summary["mean"] = per_fold.mean(axis=1)
    summary["std_across_folds"] = per_fold.std(axis=1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(out_dir / "fold_auroc_per_run.csv", index=False)
    summary.round(4).to_csv(out_dir / "folds_summary.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    print(summary.round(3).to_string())
    print(f"\nSaved {out_dir / 'folds_summary.csv'} and {out_dir / 'fold_auroc_per_run.csv'}")


if __name__ == "__main__":
    main()