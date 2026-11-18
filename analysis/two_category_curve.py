"""One figure with two panels (screw, metal_nut): PatchCore, WinCLIP and YOLO versus the number of images N.

Input : results/few_shot.csv, results/winclip_fewshot.csv   (screw: PatchCore with N good images, WinCLIP)
        results_fewshot_3cat.csv                            (metal_nut: PatchCore and WinCLIP; repo root or results/)
        results_15.csv                                      (PatchCore on all training images: horizontal reference line)
        results/yolo_fewshot.csv + results/yolo_folds.csv                      (screw: YOLO, N defective images)
        results/yolo_fewshot_metal_nut.csv + results/yolo_folds_metal_nut.csv  (metal_nut: YOLO)
Output: figures/two_category_curve.png, results/two_category_curve.csv

X axis: N means GOOD images for PatchCore and WinCLIP (the reference images) but DEFECTIVE images for YOLO.
The three curves share an axis only for orientation. Do not read them point against point.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
ORANGE, BLUE, VIOLET, REF = "#d9822b", "#2a78d6", "#6f42c1", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
CATS = ["screw", "metal_nut"]
ZERO_X = 0.6


def find(name):
    return next(p for p in (ROOT / name, RES / name) if p.exists())


def stats(df, col, by):
    g = df.groupby(by)[col]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1).fillna(0.0), "n_runs": g.size()})


def anomaly_runs():
    """PatchCore / WinCLIP few-shot runs of both categories in one table (columns: category, model, coreset_ratio, n_train, image_AUROC)."""
    cols = ["category", "model", "coreset_ratio", "n_train", "image_AUROC"]
    parts = [pd.read_csv(RES / "few_shot.csv"), pd.read_csv(RES / "winclip_fewshot.csv"), pd.read_csv(find("results_fewshot_3cat.csv"))]
    return pd.concat([p[cols] for p in parts], ignore_index=True)


def patchcore_few(runs, cat):
    d = runs[(runs.category == cat) & (runs.model == "patchcore") & (runs.coreset_ratio == 1.0) & (runs.n_train <= 50)]
    return stats(d, "image_AUROC", "n_train")


def winclip_few(runs, cat):
    d = runs[(runs.category == cat) & (runs.model == "winclip")]
    return stats(d, "image_AUROC", "n_train")


def patchcore_full(cat):
    d = pd.read_csv(find("results_15.csv"))
    d = d[(d.category == cat) & (d.model == "patchcore")]
    return float(d.groupby("seed")["image_AUROC"].mean().mean())


def yolo_curve(cat):
    """Mean over the 3 draws inside each fold, then mean and std (n - 1) over the 4 folds. Last row: all defective images."""
    sfx = "" if cat == "screw" else f"_{cat}"
    few = pd.read_csv(RES / f"yolo_fewshot{sfx}.csv")
    per_fold = few.groupby(["n_defect", "fold"])["image_AUROC"].mean().unstack("fold")
    out = pd.DataFrame({"mean": per_fold.mean(axis=1), "std": per_fold.std(axis=1), "n_runs": few.groupby("n_defect").size()})
    out.index = out.index.astype(float)
    full = pd.read_csv(RES / f"yolo_folds{sfx}.csv")
    full = full[full["imgsz"] == 640].sort_values("fold")["image_AUROC"].to_numpy()
    assert len(full) == 4, f"{cat}: expected 4 full-data YOLO runs, found {len(full)}"
    folds_file = ROOT / "splits" / f"{cat}_folds.csv"
    if folds_file.exists():
        f = pd.read_csv(folds_file)
        n_pool = float(round(np.mean([((f["fold"] != k) & (f["label"] == 1)).sum() for k in sorted(f["fold"].unique())])))
    else:
        n_pool = float(out.index.max() * 1.5)
    out.loc[n_pool] = [full.mean(), full.std(ddof=1), 4]
    return out, n_pool


def draw_panel(ax, cat, runs, show_ylabel):
    pc, wc, yo, n_pool = patchcore_few(runs, cat), winclip_few(runs, cat), None, None
    yo, n_pool = yolo_curve(cat)
    pc_full = patchcore_full(cat)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)
    ax.axhline(pc_full, color=REF, lw=1, ls=(0, (4, 3)))
    ax.text(0.62, pc_full + 0.006, f"PatchCore, all good training images: {pc_full:.3f}", fontsize=8, color=INK2)
    ax.axhline(0.5, color=REF, lw=1, ls=(0, (1, 3)))
    ax.text(n_pool * 1.25, 0.505, "chance", ha="right", fontsize=8, color=INK2, va="bottom")

    tidy = []
    for df, color, name, dodge in ((pc, ORANGE, "PatchCore: N GOOD images", 0.92),
                                   (wc, VIOLET, "WinCLIP: 0 = zero-shot, N = reference GOOD images", 1.0),
                                   (yo, BLUE, "YOLOv8s (supervised): N DEFECTIVE images", 1.08)):
        x = np.array([ZERO_X if (name.startswith("WinCLIP") and n == 0) else n * dodge for n in df.index], dtype=float)
        m, s = df["mean"].to_numpy(), df["std"].to_numpy()
        order = np.argsort(x)
        ax.fill_between(x[order], (m - s)[order], (m + s)[order], color=color, alpha=0.12, lw=0, zorder=2)
        ax.plot(x[order], m[order], color=color, lw=2, zorder=4)
        ax.errorbar(x, m, yerr=s, fmt="o", color=color, ms=6, capsize=3, lw=1.3, mec=SURFACE, mew=1.2, zorder=5, label=name)
        method = name.split(":")[0].split(" (")[0]
        for n, row in df.iterrows():
            tidy.append([cat, method, "all" if (method == "YOLOv8s" and n == n_pool) else int(n), round(row["mean"], 4),
                         round(row["std"], 4), int(row["n_runs"])])

    ticks = sorted(set([ZERO_X] + [float(n) for n in pc.index] + [float(n) for n in yo.index if n != n_pool]
                       + [float(n) for n in wc.index if n > 0]) | {n_pool})
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0" if t == ZERO_X else ("all" if t == n_pool else f"{t:g}") for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.5, n_pool * 1.3), ax.set_ylim(0.45, 1.03)
    ax.set_xlabel("N (good images for PatchCore / WinCLIP, defective for YOLO)", fontsize=8.5)
    if show_ylabel:
        ax.set_ylabel("image AUROC", fontsize=9)
    ax.set_title(cat.replace("_", " "), loc="left", fontsize=11.5, color=INK, fontweight="bold")
    return tidy


def main():
    runs = anomaly_runs()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), facecolor=SURFACE, sharey=True)
    rows = []
    for i, cat in enumerate(CATS):
        axes[i].set_facecolor(SURFACE)
        rows += draw_panel(axes[i], cat, runs, show_ylabel=(i == 0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, "Bars/bands = std (PatchCore and WinCLIP: over 3 draws; YOLO: over 4 folds, each fold = mean of 3 draws). "
             "N counts GOOD images for PatchCore and WinCLIP but DEFECTIVE images for YOLO:\nthe curves are for orientation only, "
             "not point-by-point comparable. YOLO is scored on held-out folds, the other methods on the whole test set.",
             fontsize=7.6, color=INK2, va="bottom")
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    fig.savefig(FIG / "two_category_curve.png", dpi=150, facecolor=SURFACE)

    t = pd.DataFrame(rows, columns=["category", "method", "N", "image_AUROC_mean", "std", "n_runs"])
    t.to_csv(RES / "two_category_curve.csv", index=False)
    pd.set_option("display.width", 160)
    print(t.to_string(index=False))
    print("Saved", FIG / "two_category_curve.png", "and", RES / "two_category_curve.csv")


if __name__ == "__main__":
    main()
