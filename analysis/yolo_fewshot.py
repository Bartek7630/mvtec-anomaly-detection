"""Summarise the YOLO few-shot experiment and draw the learning curve. Works for any category (--category, default screw).

Input : results/yolo_fewshot.csv   (one row per training: fold, n_defect, train_seed, image/pixel AUROC)
        results/yolo_folds.csv     (the full-data YOLO runs: used as the last point of the curve)
        results/folds_summary.csv  (PatchCore per fold: used as the reference line)
        For every category except screw the file names get the suffix _<category>:
        results/yolo_fewshot_metal_nut.csv, results/yolo_folds_metal_nut.csv, results/folds_summary_metal_nut.csv
Output: results/yolo_fewshot_summary[_<category>].csv, figures/yolo_fewshot_curve[_<category>].png

    python analysis/yolo_fewshot.py
    python analysis/yolo_fewshot.py --category metal_nut
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, REF = "#2a78d6", "#d9822b", "#8a8984"


def summarise(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean over seeds inside each fold, then mean and std (n - 1) across folds, for every N."""
    per_fold = df.groupby(["n_defect", "fold"])[metric].mean().unstack("fold")
    out = pd.DataFrame({"mean": per_fold.mean(axis=1), "std_across_folds": per_fold.std(axis=1)})
    out["min_fold"], out["max_fold"] = per_fold.min(axis=1), per_fold.max(axis=1)
    seed_std = df.groupby(["n_defect", "fold"])[metric].std().groupby("n_defect").mean()
    out["mean_std_across_seeds"] = seed_std
    return out.join(per_fold.add_prefix("fold_"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="screw")
    category = parser.parse_args().category
    sfx = "" if category == "screw" else f"_{category}"
    pretty = category.replace("_", " ")

    few = pd.read_csv(RES / f"yolo_fewshot{sfx}.csv")
    assert len(few) == few[["fold", "n_defect", "train_seed"]].drop_duplicates().shape[0], "duplicate runs in the CSV"
    n_runs = few.groupby("n_defect").size()
    print("runs per N:", n_runs.to_dict())

    image = summarise(few, "image_AUROC")
    pixel = summarise(few, "pixel_AUROC")

    full = pd.read_csv(RES / f"yolo_folds{sfx}.csv")
    full = full[full["imgsz"] == 640].sort_values("fold")
    full_img = full["image_AUROC"].to_numpy()
    assert len(full_img) == 4, f"expected 4 full-data YOLO runs (imgsz 640), found {len(full_img)}"
    summary = pd.read_csv(RES / f"folds_summary{sfx}.csv").set_index("method")
    patchcore = summary[summary.index.str.contains("patchcore") & summary.index.str.contains("cr0.01_nall")].iloc[0]
    pc_mean = float(patchcore["mean"])
    folds_file = ROOT / "splits" / f"{category}_folds.csv"
    n_pool = None
    if folds_file.exists():
        f = pd.read_csv(folds_file)
        n_pool = int(round(np.mean([((f["fold"] != k) & (f["label"] == 1)).sum() for k in sorted(f["fold"].unique())])))
    all_tag = f"all\n(~{n_pool})" if n_pool else "all"

    table = image.round(4).assign(pixel_AUROC_mean=pixel["mean"].round(4))
    table.loc["all (1 seed)"] = [round(full_img.mean(), 4), round(full_img.std(ddof=1), 4), full_img.min(), full_img.max(),
                                      np.nan, *full_img, round(full["pixel_AUROC"].mean(), 4)]
    table.index.name = "n_defect"
    table.to_csv(RES / f"yolo_fewshot_summary{sfx}.csv")
    pd.set_option("display.width", 200)
    print(table.to_string())

    ns = list(image.index)
    xs = np.arange(len(ns) + 1)
    labels = [str(n) for n in ns] + [all_tag]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID), ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)

    for i, n in enumerate(ns):
        sub = few[few.n_defect == n]
        jitter = (sub["fold"].to_numpy() - 1.5) * 0.07
        ax.scatter(np.full(len(sub), i) + jitter, sub["image_AUROC"], s=12, color=BLUE, alpha=0.35, linewidths=0, zorder=3)
    for j, v in enumerate(full_img):
        ax.scatter(len(ns) + (j - 1.5) * 0.07, v, s=12, color=ORANGE, alpha=0.5, linewidths=0, zorder=3)

    mu, sd = image["mean"].to_numpy(), image["std_across_folds"].to_numpy()
    ax.plot(xs[:-1], mu, color=BLUE, lw=2, zorder=4)
    ax.errorbar(xs[:-1], mu, yerr=sd, fmt="o", color=BLUE, ms=8, capsize=3, lw=1.5, mec=SURFACE, mew=1.5, zorder=5,
                label="YOLOv8s, 640 px (mean of 4 folds; bars = std across folds)")
    ax.errorbar([xs[-1]], [full_img.mean()], yerr=[full_img.std(ddof=1)], fmt="o", color=ORANGE, mfc=SURFACE, ms=8,
                capsize=3, lw=1.5, mew=2, zorder=5, label="YOLOv8s, all defective images (1 training per fold)")
    ax.plot([xs[-2], xs[-1]], [mu[-1], full_img.mean()], color=BLUE, lw=1.5, ls=(0, (3, 2)), zorder=4)
    for x, v in zip(xs[:-1], mu):
        ax.annotate(f"{v:.3f}", (x, v), xytext=(9, -14), textcoords="offset points", fontsize=9, color=INK)
    ax.annotate(f"{full_img.mean():.3f}", (xs[-1], full_img.mean()), xytext=(9, -14), textcoords="offset points",
                fontsize=9, color=INK)

    ax.axhline(pc_mean, color=REF, lw=1, ls=(0, (4, 3)))
    ax.text(-0.35, pc_mean + 0.006, f"PatchCore, all good training images: {pc_mean:.3f}", ha="left", fontsize=8.5, color=INK2)
    ax.axhline(0.5, color=REF, lw=1, ls=(0, (1, 3)))
    ax.text(xs[-1] + 0.45, 0.505, "chance = 0.5", ha="right", fontsize=8.5, color=INK2, va="bottom")

    ax.set_xticks(xs), ax.set_xticklabels(labels)
    ax.set_xlim(-0.45, xs[-1] + 0.5), ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("number of DEFECTIVE training images N (plus as many good ones as the pool has)", fontsize=9, color=INK2)
    ax.set_ylabel("image AUROC", fontsize=9, color=INK2)
    ax.tick_params(colors=INK2)
    ax.set_title(f"YOLOv8 on MVTec AD '{pretty}': how many defective images are needed?", loc="left", fontsize=11.5, color=INK, pad=34)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left", bbox_to_anchor=(0, 1.13), ncol=1)
    fig.text(0.01, 0.012, "Small dots: single trainings (3 seeds x 4 folds per N). About 1500 training iterations for every N.\n"
             "N means defective images for YOLO, but good images for PatchCore.", fontsize=7.6, color=INK2, va="bottom")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(FIG / f"yolo_fewshot_curve{sfx}.png", dpi=150, facecolor=SURFACE)
    print("Saved", RES / f"yolo_fewshot_summary{sfx}.csv", "and", FIG / f"yolo_fewshot_curve{sfx}.png")


if __name__ == "__main__":
    main()
