"""One figure that answers the thesis question - anomaly detection vs supervised YOLO.

Input : results/few_shot.csv           PatchCore, N GOOD images (seeds 1, 2, 42), test = all 160 test images
        results/stfpm_fewshot.csv      STFPM, N GOOD images (same seeds and draws), 2000 training steps
        results/stfpm_screw.csv        STFPM trained on all 320 good images (100 epochs = 2000 steps)
        results/efficientad_fewshot.csv  EfficientAD, N = 50 good images, ONE draw (seed 42), 70 000 steps
        results/efficientad_screw.csv    EfficientAD trained on 288 good images (the other 10% = validation), ONE run
        results/winclip_fewshot.csv    WinCLIP (optional): N = 0 (zero-shot) and N = 1, 5, 10 reference GOOD images (seeds 1, 2, 42)
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
BLUE, ORANGE, GREY, TEAL, MAGENTA, VIOLET = "#2a78d6", "#d9822b", "#8a8984", "#2a9d8f", "#a23b72", "#6f42c1"


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


def efficientad_points() -> pd.DataFrame:
    """EfficientAD: single runs only (seed 42), so there is no std. N = 50 and the full run (288 good images + 32 held out)."""
    few = pd.read_csv(RES / "efficientad_fewshot.csv")
    full = pd.read_csv(RES / "efficientad_screw.csv")
    df = pd.concat([few[few["n_train"] == 50], full[full["n_train"] > 100]])
    assert len(df) == 2 and (df["seed"] == 42).all(), df
    out = pd.DataFrame({"mean": df["image_AUROC"].to_numpy(), "std": np.nan, "n_runs": 1}, index=df["n_train"].to_numpy())
    out.index.name = "n_train"
    return out


def winclip_curve():
    """WinCLIP (optional): N = 0 is zero-shot (one run, nothing is drawn from the training set), N > 0 uses N reference
    good images (same seeds and draws as PatchCore). Returns None if results/winclip_fewshot.csv does not exist yet."""
    path = RES / "winclip_fewshot.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["model"] == "winclip"]
    g = df.groupby("n_train")["image_AUROC"]
    out = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1).fillna(0.0), "n_runs": g.count()})
    out.index.name = "n_train"
    return out


def yolo_curve() -> pd.DataFrame:
    """Mean over seeds inside each fold, then mean and std (n - 1) across the 4 folds; plus the all-data point."""
    few = pd.read_csv(RES / "yolo_fewshot.csv")
    per_fold = few.groupby(["n_defect", "fold"])["image_AUROC"].mean().unstack("fold")
    out = pd.DataFrame({"mean": per_fold.mean(axis=1), "std": per_fold.std(axis=1), "n_runs": few.groupby("n_defect").size()})
    full = pd.read_csv(RES / "yolo_folds.csv")
    full = full[full["imgsz"] == 640].sort_values("fold")["image_AUROC"].to_numpy()
    assert len(full) == 4
    out.loc[89] = [full.mean(), full.std(ddof=1), 4]
    return out


def main():
    pc, yo, st, ea, wc = patchcore_curve(), yolo_curve(), stfpm_curve(), efficientad_points(), winclip_curve()
    padim = pd.read_csv(RES / "folds_summary.csv").set_index("method").filter(like="padim", axis=0).iloc[0]
    padim_mean = float(padim["mean"])

    table = pd.concat([pc.assign(method="PatchCore (N good images)"), st.assign(method="STFPM (N good images)"),
                       ea.assign(method="EfficientAD (N good images, 1 run)"),
                       yo.assign(method="YOLOv8s (N defective images)")]
                      + ([wc.assign(method="WinCLIP (N=0: zero-shot; N>0: reference good images)")] if wc is not None else []))
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

    ax.axhline(padim_mean, color=GREY, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(1.05, padim_mean + 0.008, f"PaDiM, all 320 good images: {padim_mean:.3f}", fontsize=8.5, color=INK2)
    ax.axhline(0.5, color=GREY, lw=1, ls=(0, (1, 3)), zorder=2)
    ax.text(320, 0.505, "chance = 0.5", ha="right", fontsize=8.5, color=INK2, va="bottom")

    for df, color, name, dodge in ((pc, ORANGE, "PatchCore (anomaly detection): N GOOD images", 0.9),
                                   (st, TEAL, "STFPM (anomaly detection, trained): N GOOD images", 1.0),
                                   (yo, BLUE, "YOLOv8s (supervised): N DEFECTIVE images", 1.1)):
        x, m, s = df.index.to_numpy(float) * dodge, df["mean"].to_numpy(), df["std"].to_numpy()
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.13, lw=0, zorder=2)
        ax.plot(x, m, color=color, lw=2, zorder=4)
        ax.errorbar(x, m, yerr=s, fmt="o", color=color, ms=7, capsize=3, lw=1.4, mec=SURFACE, mew=1.4, zorder=5, label=name)

    xe, me = ea.index.to_numpy(float) * 1.06, ea["mean"].to_numpy()
    ax.plot(xe, me, color=MAGENTA, lw=1.4, ls=(0, (5, 2)), zorder=4)
    ax.plot(xe, me, "D", color=MAGENTA, ms=8, mec=SURFACE, mew=1.4, zorder=5,
            label="EfficientAD (anomaly detection, trained): N GOOD images, 1 run (no error bar)")

    if wc is not None:
        few = wc[wc.index > 0]
        if len(few):
            x, m, s_ = few.index.to_numpy(float) * 0.8, few["mean"].to_numpy(), few["std"].to_numpy()
            ax.fill_between(x, m - s_, m + s_, color=VIOLET, alpha=0.13, lw=0, zorder=2)
            ax.plot(x, m, color=VIOLET, lw=2, zorder=4)
            ax.errorbar(x, m, yerr=s_, fmt="^", color=VIOLET, ms=8, capsize=3, lw=1.4, mec=SURFACE, mew=1.4, zorder=5,
                        label="WinCLIP (text + reference images, no training): N GOOD images")
        if 0 in wc.index:
            z = float(wc.loc[0, "mean"])
            ax.axhline(z, color=VIOLET, lw=1.0, ls=(0, (1, 3)), zorder=2)
            ax.plot([0.6], [z], "^", color=VIOLET, ms=9, mec=SURFACE, mew=1.4, zorder=6,
                    label=None if len(wc[wc.index > 0]) else "WinCLIP zero-shot (N = 0)")
            ax.annotate(f"zero-shot {z:.3f}", (0.6, z), xytext=(0, 9), textcoords="offset points", fontsize=8.5, color=INK, ha="left")

    ax.annotate(f"EfficientAD {me[-1]:.3f}", (xe[-1], me[-1]), xytext=(10, 9), textcoords="offset points",
                fontsize=9, color=INK, va="center")
    ax.annotate(f"{pc['mean'].iloc[-1]:.3f}", (320 * 0.9, pc["mean"].iloc[-1]), xytext=(8, -9), textcoords="offset points",
                fontsize=9, color=INK, va="center")
    ax.annotate(f"{st['mean'].iloc[-1]:.3f}", (320, st["mean"].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                fontsize=9, color=INK, va="center")
    ax.annotate(f"{yo['mean'].iloc[-1]:.3f}", (89 * 1.1, yo["mean"].iloc[-1]), xytext=(8, 8), textcoords="offset points",
                fontsize=9, color=INK, va="center")

    ax.set_xscale("log")
    ticks = [1, 5, 10, 20, 50, 89, 320] if wc is None else [0.6, 1, 5, 10, 20, 50, 89, 320]
    ax.set_xticks(ticks), ax.set_xticklabels(["0" if t == 0.6 else ("~89" if t == 89 else str(t)) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.8 if wc is None else 0.45, 480), ax.set_ylim(0.35, 1.03)
    ax.set_xlabel("N = number of training images (log scale) - GOOD for PatchCore, DEFECTIVE for YOLO", fontsize=9, color=INK2)
    ax.set_ylabel("image AUROC", fontsize=9, color=INK2)
    ax.tick_params(colors=INK2)
    ax.set_title("MVTec AD 'screw': anomaly detection vs supervised detection", loc="left", fontsize=12, color=INK, pad=62)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.27))
    fig.text(0.01, 0.012,
             "The X axis counts two DIFFERENT kinds of images: good ones for PatchCore, defective ones for YOLO (YOLO also sees up to ~30 good ones).\n"
             "Same 160 test images for both: PatchCore scores them with one model, YOLO fold by fold (4 models x 40 held-out images, AUROC averaged).\n"
             "Points = mean, bars/bands = std (PatchCore, STFPM: 3 draws of the training images; YOLO: 4 folds); curves are shifted sideways for readability.\n"
             "PatchCore and STFPM use the same protocol and the same draws, so they can be compared point by point; YOLO cannot.\n" +
             ("WinCLIP: N = 0 is zero-shot (no training images; one run), N > 0 = reference images (3 draws, the same as PatchCore).\n" if wc is not None else "") +
             "EfficientAD: one run only (seed 42); N = 50 uses the test set for normalisation statistics, the full run (288 images) a 10% validation split.",
             fontsize=7.4, color=INK2, va="bottom")
    fig.tight_layout(rect=[0, 0.15 if wc is None else 0.18, 1, 1])
    fig.savefig(FIG / "anomaly_vs_supervised.png", dpi=150, facecolor=SURFACE)
    print("Saved", RES / "anomaly_vs_supervised.csv", "and", FIG / "anomaly_vs_supervised.png")


if __name__ == "__main__":
    main()