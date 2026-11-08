"""One figure that answers the thesis question - anomaly detection vs supervised YOLO.

Input : results/few_shot.csv           PatchCore, N GOOD images (seeds 1, 2, 42), test = all 160 test images
        results/stfpm_fewshot.csv      STFPM, N GOOD images (same seeds and draws), 2000 training steps
        results/stfpm_screw.csv        STFPM trained on all 320 good images (100 epochs = 2000 steps)
        results/yolo_fewshot.csv       YOLO, N DEFECTIVE images (3 seeds x 4 folds), test = the held-out fold
        results/yolo_folds.csv         YOLO trained on all defective images (imgsz 640), 1 training per fold
        results/folds_summary.csv      PaDiM, full data, per fold (horizontal line)
Output: results/anomaly_vs_supervised.csv, figures/anomaly_vs_supervised.png

Careful: the two curves have the same X axis but N means different images (good vs defective)
and the evaluation differs (one model on all 160 images vs 4 models on 40 held-out images each). Never read the curves point against point.
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
BLUE, ORANGE, GREY, TEAL = "#2a78d6", "#d9822b", "#8a8984", "#2a9d8f"


def patchcore_curve() -> pd.DataFrame:
    """Mean and std over the 3 seeds for every N (coreset 100% for N <= 50, 1% for N = 320 as in the README)."""
    df = pd.read_csv(RES / "few_shot.csv")
    df = df[(df["n_train"] == 320) | (df["coreset_ratio"] == 1.0)]
    g = df.groupby("n_train")["image_AUROC"]
    out = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "n_runs": g.size()})
    assert (out["n_runs"] == 3).all(), out
    return out


def stfpm_curve() -> pd.DataFrame:
    """STFPM: N good images, 3 draws (seeds 42, 1, 2) as for PatchCore; N = 320 is the full-data run (100 epochs = 2000 steps)."""
    few = pd.read_csv(RES / "stfpm_fewshot.csv")
    full = pd.read_csv(RES / "stfpm_screw.csv")
    df = pd.concat([few, full[full["epochs"] == 100]])
    assert (df.loc[df["n_train"] != 320, "steps"].fillna(2000) == 2000).all(), "few-shot runs must all have 2000 steps"
    g = df.groupby("n_train")["image_AUROC"]
    out = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "n_runs": g.size()})
    assert (out["n_runs"] == 3).all(), out
    return out


def yolo_curve() -> pd.DataFrame:
    """Mean over seeds inside each fold, then mean and std (n - 1) across the 4 folds; plus the all-data point."""
    few = pd.read_csv(RES / "yolo_fewshot.csv")
    per_fold = few.groupby(["n_defect", "fold"])["image_AUROC"].mean().unstack("fold")
    out = pd.DataFrame({"mean": per_fold.mean(axis=1), "std": per_fold.std(axis=1), "n_runs": few.groupby("n_defect").size()})
    full = pd.read_csv(RES / "yolo_folds.csv")
    full = full[full["imgsz"] == 640].sort_values("fold")["image_AUROC"].to_numpy()
    assert len(full) == 4
    # training pool = defective images of the other 3 folds: 90 / 89 / 89 / 89 -> about 89
    out.loc[89] = [full.mean(), full.std(ddof=1), 4]
    return out


def main():
    pc, yo, st = patchcore_curve(), yolo_curve(), stfpm_curve()
    padim = pd.read_csv(RES / "folds_summary.csv").set_index("method").filter(like="padim", axis=0).iloc[0]
    padim_mean = float(padim["mean"])

    table = pd.concat([pc.assign(method="PatchCore (N good images)"), st.assign(method="STFPM (N good images)"),
                       yo.assign(method="YOLOv8s (N defective images)")])
    table.index.name = "N"
    table.round(4).to_csv(RES / "anomaly_vs_supervised.csv")
    print(table.round(3).to_string())
    print(f"PaDiM, full data (mean over folds): {padim_mean:.3f}")

    fig, ax = plt.subplots(figsize=(8.0, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID), ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8), ax.set_axisbelow(True)

    # PaDiM: a single value for the full data, drawn as a horizontal line
    ax.axhline(padim_mean, color=GREY, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(1.05, padim_mean + 0.008, f"PaDiM, all 320 good images: {padim_mean:.3f}", fontsize=8.5, color=INK2)
    ax.axhline(0.5, color=GREY, lw=1, ls=(0, (1, 3)), zorder=2)
    ax.text(320, 0.505, "chance = 0.5", ha="right", fontsize=8.5, color=INK2, va="bottom")

    # the three curves are shifted slightly sideways (in log space) so that their error bars do not overlap
    for df, color, name, dodge in ((pc, ORANGE, "PatchCore (anomaly detection): N GOOD images", 0.9),
                                   (st, TEAL, "STFPM (anomaly detection, trained): N GOOD images", 1.0),
                                   (yo, BLUE, "YOLOv8s (supervised): N DEFECTIVE images", 1.1)):
        x, m, s = df.index.to_numpy(float) * dodge, df["mean"].to_numpy(), df["std"].to_numpy()
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.13, lw=0, zorder=2)
        ax.plot(x, m, color=color, lw=2, zorder=4)
        ax.errorbar(x, m, yerr=s, fmt="o", color=color, ms=7, capsize=3, lw=1.4, mec=SURFACE, mew=1.4, zorder=5, label=name)

    # direct labels at the right end of each curve
    ax.annotate(f"{pc['mean'].iloc[-1]:.3f}", (320 * 0.9, pc["mean"].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                fontsize=9, color=INK, va="center")
    ax.annotate(f"{st['mean'].iloc[-1]:.3f}", (320, st["mean"].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                fontsize=9, color=INK, va="center")
    ax.annotate(f"{yo['mean'].iloc[-1]:.3f}", (89 * 1.1, yo["mean"].iloc[-1]), xytext=(8, 8), textcoords="offset points",
                fontsize=9, color=INK, va="center")

    ax.set_xscale("log")
    ticks = [1, 5, 10, 20, 50, 89, 320]
    ax.set_xticks(ticks), ax.set_xticklabels([str(t) if t != 89 else "~89" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.8, 480), ax.set_ylim(0.35, 1.03)
    ax.set_xlabel("N = number of training images (log scale) - GOOD for PatchCore, DEFECTIVE for YOLO", fontsize=9, color=INK2)
    ax.set_ylabel("image AUROC", fontsize=9, color=INK2)
    ax.tick_params(colors=INK2)
    ax.set_title("MVTec AD 'screw': anomaly detection vs supervised detection", loc="left", fontsize=12, color=INK, pad=48)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.2))
    fig.text(0.01, 0.012,
             "The X axis counts two DIFFERENT kinds of images: good ones for PatchCore, defective ones for YOLO (YOLO also sees up to ~30 good ones).\n"
             "Same 160 test images for both: PatchCore scores them with one model, YOLO fold by fold (4 models x 40 held-out images, AUROC averaged).\n"
             "Points = mean, bars/bands = std (PatchCore, STFPM: 3 draws of the training images; YOLO: 4 folds); curves are shifted sideways for readability.\n"
             "PatchCore and STFPM use the same protocol and the same draws, so they can be compared point by point; YOLO cannot.",
             fontsize=7.4, color=INK2, va="bottom")
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(FIG / "anomaly_vs_supervised.png", dpi=150, facecolor=SURFACE)
    print("Saved", RES / "anomaly_vs_supervised.csv", "and", FIG / "anomaly_vs_supervised.png")


if __name__ == "__main__":
    main()