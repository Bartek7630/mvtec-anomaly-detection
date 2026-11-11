"""Which defect types are hard for which method?

Input : scores/*.csv            one file per run (columns path, label, score), name ends with _seed<N>
        splits/screw_folds.csv  (columns path, defect_type, label, fold)
        --root                  MVTec AD category folder (only needed for the pictures of missed images)
Output: results/auroc_by_defect_type.csv   AUROC "good vs one defect type" for every method
        results/recall_by_defect_type.csv  share of defective images found at a threshold that keeps 95% of good images
        results/missed_<method>.csv        the defective images with the lowest scores
        figures/auroc_by_defect_type.png   bar chart
        figures/missed_<method>.png        the most missed images with their ground-truth mask (needs --root)


"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT_REPO = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLORS = {"patchcore": "#d9822b", "yolo": "#2a78d6", "padim": "#8a8984", "stfpm": "#2a9d8f", "efficientad": "#a23b72"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help="MVTec AD category folder (for the pictures of missed images)")
    p.add_argument("--scores-dir", default=str(ROOT_REPO / "scores"))
    p.add_argument("--folds", default=str(ROOT_REPO / "splits" / "screw_folds.csv"))
    p.add_argument("--methods", nargs="+", default=["patchcore", "padim", "stfpm", "efficientad", "yolov8s_img640"],
                   help="a score file is used if its method name contains one of these words")
    p.add_argument("--good-kept", type=float, default=0.95, help="share of good images that must stay below the threshold")
    p.add_argument("--n-missed", type=int, default=2, help="how many most-missed images to draw per method")
    return p.parse_args()


def load_runs(scores_dir: Path, folds: pd.DataFrame, words: list) -> pd.DataFrame:
    """Stack all chosen score files into one table: method, seed, path, label, score + defect_type, fold."""
    tables = []
    for file in sorted(scores_dir.glob("*.csv")):
        method, _, seed = file.stem.rpartition("_seed")
        if not method or not any(w in method for w in words):
            continue
        t = pd.read_csv(file)
        assert len(t) == len(folds), f"{file.name}: {len(t)} rows, expected {len(folds)}"
        t["method"], t["seed"] = method, int(seed)
        tables.append(t)
    if not tables:
        raise SystemExit(f"No score file in {scores_dir} matches {words}")
    runs = pd.concat(tables, ignore_index=True)
    runs = runs.merge(folds[["path", "defect_type", "fold"]], on="path", how="left", validate="many_to_one")
    assert runs["defect_type"].notna().all(), "some paths in the score files are not in the folds file"
    return runs


def short_name(method: str) -> str:
    """screw_patchcore_wide_resnet50_2_... -> patchcore; screw_yolov8s_img640_pool -> yolo 640 px."""
    if "patchcore" in method:
        return "patchcore"
    if "padim" in method:
        return "padim"
    if "stfpm" in method:
        return "stfpm"
    if "efficientad" in method:
        return "efficientad"
    if "yolo" in method:
        return "yolo " + method.split("_img")[-1].split("_")[0] + " px"
    return method


def color_of(name: str) -> str:
    return next(c for key, c in COLORS.items() if name.startswith(key))


def auroc_by_type(runs: pd.DataFrame, types: list) -> pd.DataFrame:
    """AUROC of 'all good images vs the images of one defect type', separately for every run."""
    rows = []
    for (method, seed), run in runs.groupby(["method", "seed"]):
        good = run[run["defect_type"] == "good"]
        for t in types + ["ALL DEFECTS"]:
            bad = run[run["defect_type"] != "good"] if t == "ALL DEFECTS" else run[run["defect_type"] == t]
            both = pd.concat([good, bad])
            rows.append({"method": method, "seed": seed, "defect_type": t, "n_defective": len(bad),
                         "auroc": roc_auc_score(both["label"], both["score"])})
    per_run = pd.DataFrame(rows)
    out = (per_run.groupby(["method", "defect_type"])
           .agg(n_defective=("n_defective", "first"), auroc=("auroc", "mean"), std_seeds=("auroc", "std"),
                n_seeds=("seed", "nunique"))
           .reset_index())
    return out


def mean_scores(runs: pd.DataFrame) -> pd.DataFrame:
    """One score per image and method: the mean over seeds (for a single seed it is that seed)."""
    return runs.groupby(["method", "path", "defect_type", "label"], as_index=False)["score"].mean()


def recall_and_missed(scores: pd.DataFrame, types: list, good_kept: float):
    """Threshold = the score below which `good_kept` of the good images lie; a defective image with a score
    at or below it is MISSED. Returns recall per defect type and the list of missed images."""
    recall_rows, missed = [], {}
    for method, m in scores.groupby("method"):
        good = m[m["defect_type"] == "good"]["score"].to_numpy()
        thr = float(np.quantile(good, good_kept))
        bad = m[m["label"] == 1].copy()
        bad["found"] = bad["score"] > thr
        for t in types:
            sub = bad[bad["defect_type"] == t]
            recall_rows.append({"method": method, "defect_type": t, "n_defective": len(sub),
                                "found": int(sub["found"].sum()), "recall": sub["found"].mean(), "threshold": thr})
        missed[method] = bad[~bad["found"]].sort_values(["score", "path"]).assign(threshold=thr)
    return pd.DataFrame(recall_rows), missed


def plot_auroc(table: pd.DataFrame, types: list, out: Path):
    methods = list(table["method"].unique())
    names = [short_name(m) for m in methods]
    x = np.arange(len(types) + 1)
    labels = [f"{t}\n(n={int(table[(table.defect_type == t)]['n_defective'].iloc[0])})" for t in types]
    labels.append(f"all defects\n(n={int(table[table.defect_type == 'ALL DEFECTS']['n_defective'].iloc[0])})")
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9.2 + 0.9 * max(0, len(methods) - 4), 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID), ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)
    for i, (m, name) in enumerate(zip(methods, names)):
        sub = table[table["method"] == m].set_index("defect_type").loc[types + ["ALL DEFECTS"]]
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, sub["auroc"], width=width * 0.9, color=color_of(name), label=name, zorder=3)
        ax.errorbar(pos, sub["auroc"], yerr=sub["std_seeds"].fillna(0), fmt="none", ecolor=INK2, capsize=2, lw=1, zorder=4)
        for xp, v, s in zip(pos, sub["auroc"], sub["std_seeds"].fillna(0)):
            ax.text(xp, v + s + 0.012, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.axhline(0.5, color="#8a8984", lw=1, ls=(0, (1, 3)))
    ax.set_xticks(x), ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0.4, 1.08), ax.set_ylabel("image AUROC: good vs this defect type", fontsize=9, color=INK2)
    ax.tick_params(colors=INK2)
    ax.set_title("Which defect types are hard?", loc="left", fontsize=12, color=INK, pad=30)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.12), ncol=len(methods))
    fig.text(0.01, 0.012, "All good test images against the defective images of one type. Error bars: std over seeds (none for YOLO: 1 training per fold).\n"
             "YOLO scores come from 4 different models (one per fold) pooled together.", fontsize=7.4, color=INK2, va="bottom")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def load_gray(path: Path) -> np.ndarray:
    from PIL import Image
    arr = np.array(Image.open(path))
    return arr[..., 0] if arr.ndim == 3 else arr


def plot_missed(method: str, missed: pd.DataFrame, root: Path, n: int, out: Path):
    show = missed.head(n)
    if show.empty:
        print(f"  {short_name(method)}: nothing missed at this threshold")
        return
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 4.6), facecolor=SURFACE, squeeze=False)
    for ax, row in zip(axes[0], show.itertuples()):
        ax.imshow(load_gray(root / row.path), cmap="gray")
        mask_file = root / "ground_truth" / row.defect_type / f"{Path(row.path).stem}_mask.png"
        if mask_file.exists():
            ax.contour(load_gray(mask_file) > 0, levels=[0.5], colors="#ff3b30", linewidths=1.2)
        ax.set_title(f"{row.defect_type}/{Path(row.path).name}\nscore {row.score:.3f}  (threshold {row.threshold:.3f})",
                     fontsize=8.5, color=INK, loc="left")
        ax.axis("off")
    fig.suptitle(f"{short_name(method)}: the most missed defective images (red = true defect)", fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    args = parse_args()
    res, fig_dir = ROOT_REPO / "results", ROOT_REPO / "figures"
    res.mkdir(exist_ok=True), fig_dir.mkdir(exist_ok=True)

    folds = pd.read_csv(args.folds)
    types = sorted(t for t in folds["defect_type"].unique() if t != "good")
    print("images per type:", folds["defect_type"].value_counts().to_dict())

    runs = load_runs(Path(args.scores_dir), folds, args.methods)
    print("methods:", {m: sorted(int(s) for s in g["seed"].unique()) for m, g in runs.groupby("method")})

    # 1. AUROC per defect type
    table = auroc_by_type(runs, types)
    table.round(4).to_csv(res / "auroc_by_defect_type.csv", index=False)
    wide = table.pivot(index="defect_type", columns="method", values="auroc").loc[types + ["ALL DEFECTS"]]
    wide.columns = [short_name(c) for c in wide.columns]
    pd.set_option("display.width", 200)
    print("\nAUROC, good vs one defect type (mean over seeds):\n", wide.round(3).to_string())
    plot_auroc(table, types, fig_dir / "auroc_by_defect_type.png")

    # 2. recall at a fixed false-alarm rate + missed images
    scores = mean_scores(runs)
    recall, missed = recall_and_missed(scores, types, args.good_kept)
    recall.round(4).to_csv(res / "recall_by_defect_type.csv", index=False)
    rw = recall.pivot(index="defect_type", columns="method", values="recall")
    rw.columns = [short_name(c) for c in rw.columns]
    print(f"\nShare of defective images found when {args.good_kept:.0%} of good images are kept below the threshold:\n",
          rw.round(2).to_string())

    for method, m in missed.items():
        name = short_name(method).replace(" ", "_")
        m[["path", "defect_type", "score", "threshold"]].to_csv(res / f"missed_{name}.csv", index=False)
        print(f"\n{short_name(method)}: {len(m)} defective images missed; lowest scores:")
        print(m[["path", "score"]].head(5).to_string(index=False))
        if args.root:
            plot_missed(method, m, Path(args.root), args.n_missed, fig_dir / f"missed_{name}.png")
    if not args.root:
        print("\n(no --root given: pictures of missed images were skipped)")


if __name__ == "__main__":
    main()