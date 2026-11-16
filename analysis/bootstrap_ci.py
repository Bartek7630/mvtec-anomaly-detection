"""95% bootstrap confidence intervals for image AUROC, and paired bootstrap differences between methods.


Usage (from the repo root):
    python analysis/bootstrap_ci.py                      # all 15 categories, B = 2000
    python analysis/bootstrap_ci.py --categories screw toothbrush --B 500

Writes results/bootstrap_ci.csv and results/bootstrap_diff.csv.

How it works:
  * n test images, S seeds. For every category we draw ONE matrix of bootstrap indices idx (B, n) and use it for every
    method. Using the same images for all methods is exactly what makes the comparison paired.
  * For one replication b: take the images idx[b], compute AUROC for every seed, average over the seeds.
    The average over seeds is the same quantity as in the results tables.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]

METHODS = {
    "PatchCore": "{cat}_patchcore_wide_resnet50_2_s256_cr0.01_nall_seed*.csv",
    "PaDiM": "{cat}_padim_wide_resnet50_2_s256_d200_nall_seed*.csv",
    "WinCLIP 0": "{cat}_winclip_ViT-B-16-plus-240_s240_n0_seed*.csv",
    "WinCLIP 10": "{cat}_winclip_ViT-B-16-plus-240_s240_n10_seed*.csv",
}
PAIRS = [("PatchCore", "PaDiM"), ("PatchCore", "WinCLIP 10"), ("WinCLIP 10", "WinCLIP 0")]
CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
              "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]


def load_method(scores_dir, category, method):
    """Returns labels (n,), scores (S, n) and the paths (n,) sorted by path; None if there are no files."""
    files = sorted(Path(scores_dir).glob(METHODS[method].format(cat=category)))
    if not files:
        return None
    labels, rows, paths = None, [], None
    for f in files:
        d = pd.read_csv(f).sort_values("path").reset_index(drop=True)
        if labels is None:
            labels, paths = d["label"].to_numpy(), d["path"].to_numpy()
        else:
            assert (d["path"].to_numpy() == paths).all() and (d["label"].to_numpy() == labels).all(), f
        rows.append(d["score"].to_numpy(dtype=float))
    return labels, np.stack(rows), paths


def auroc(labels, score):
    """AUROC = P(random defective image scores higher than random good one), ties count 1/2. Returns nan if one class is missing."""
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(score)  # average ranks for ties
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_auroc(labels, scores, idx):
    """labels (n,), scores (S, n), idx (B, n) -> (B,): AUROC averaged over the seeds, one value per replication."""
    out = np.empty(len(idx))
    for b, ix in enumerate(idx):
        lab = labels[ix]
        out[b] = np.mean([auroc(lab, s[ix]) for s in scores])
    return out


def ci(values, level=95):
    """(low, high) percentile interval, ignoring replications where AUROC did not exist."""
    a = (100 - level) / 2
    return tuple(np.nanpercentile(values, [a, 100 - a]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores-dir", default=str(ROOT / "scores"))
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--categories", nargs="+", default=CATEGORIES)
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    rows_ci, rows_diff = [], []
    for cat in args.categories:
        data = {m: load_method(args.scores_dir, cat, m) for m in METHODS}
        data = {m: v for m, v in data.items() if v is not None}
        if not data:
            print(f"skip {cat}: no score files")
            continue
        first = next(iter(data.values()))
        labels, paths = first[0], first[2]
        for m, (lab, _, pth) in data.items():
            assert (pth == paths).all() and (lab == labels).all(), f"{cat}/{m}: different test images"
        n = len(labels)
        idx = rng.integers(0, n, size=(args.B, n))

        boot = {}
        for m, (_, scores, _) in data.items():
            boot[m] = bootstrap_auroc(labels, scores, idx)
            point = np.mean([auroc(labels, s) for s in scores])
            lo, hi = ci(boot[m])
            n_valid = int(np.isfinite(boot[m]).sum())
            rows_ci.append(dict(category=cat, method=m, seeds=len(scores), auroc=point, ci_low=lo, ci_high=hi,
                                n_test=n, n_defective=int(labels.sum()), valid_replications=n_valid))
        for a, b in PAIRS:
            if a in boot and b in boot:
                d = boot[a] - boot[b]
                lo, hi = ci(d)
                point = rows_ci_point(rows_ci, cat, a) - rows_ci_point(rows_ci, cat, b)
                rows_diff.append(dict(category=cat, method_a=a, method_b=b, diff=point, ci_low=lo, ci_high=hi,
                                      significant=bool(lo > 0 or hi < 0)))
        print(f"{cat}: n={n}, methods: {', '.join(data)}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_ci, df_diff = pd.DataFrame(rows_ci), pd.DataFrame(rows_diff)
    df_ci.round(4).to_csv(out / "bootstrap_ci.csv", index=False)
    df_diff.round(4).to_csv(out / "bootstrap_diff.csv", index=False)
    print(df_diff.round(3).to_string(index=False))


def rows_ci_point(rows, cat, method):
    return next(r["auroc"] for r in rows if r["category"] == cat and r["method"] == method)


if __name__ == "__main__":
    main()
