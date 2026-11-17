"""Few-shot curves on metal_nut, transistor and carpet: PatchCore (coreset 1.0) and WinCLIP vs number of good images N.

Input : results_fewshot_3cat.csv (repo root or results/), results_15.csv (full-data PatchCore as a reference line)
Output: figures/fewshot_3cat.png (3 columns = categories; top row image AUROC, bottom row pixel AUPRO)
        results/fewshot_3cat_table.csv and results/fewshot_3cat_table.md
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
BLUE, VIOLET, REF = "#2a78d6", "#7a4fd1", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
CATS = ["metal_nut", "transistor", "carpet"]
NS = [1, 5, 10, 50]


def find(name):
    return next(p for p in (ROOT / name, RES / name) if p.exists())


def main():
    d = pd.read_csv(find("results_fewshot_3cat.csv"))
    full = pd.read_csv(find("results_15.csv"))
    full = full[full.model == "patchcore"].groupby(["category", "seed"])[["image_AUROC", "pixel_AUPRO"]].mean() \
        .groupby("category").mean()
    g = d.groupby(["category", "model", "n_train"])[["image_AUROC", "pixel_AUPRO"]].agg(["mean", "std"])

    rows = []
    for cat in CATS:
        for model, label in (("patchcore", "PatchCore"), ("winclip", "WinCLIP")):
            for n in ([0] if model == "winclip" else []) + NS:
                k = (cat, model, n)
                if k in g.index:
                    r = g.loc[k]
                    rows.append([cat, label, n,
                                 round(r[("image_AUROC", "mean")], 4), round(r[("image_AUROC", "std")], 4),
                                 round(r[("pixel_AUPRO", "mean")], 4), round(r[("pixel_AUPRO", "std")], 4)])
    t = pd.DataFrame(rows, columns=["category", "method", "N", "image_AUROC", "image_AUROC_std",
                                    "pixel_AUPRO", "pixel_AUPRO_std"])
    t.to_csv(RES / "fewshot_3cat_table.csv", index=False)

    md = ["| category | method | N | image AUROC | pixel AUPRO |", "|---|---|---|---|---|"]
    for r in t.itertuples():
        sd = lambda v: f" ± {v:.3f}" if pd.notna(v) else ""
        md.append(f"| {r.category} | {r.method} | {r.N} | {r.image_AUROC:.3f}{sd(r.image_AUROC_std)} | "
                  f"{r.pixel_AUPRO:.3f}{sd(r.pixel_AUPRO_std)} |")
    (RES / "fewshot_3cat_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), facecolor=SURFACE)
    for j, cat in enumerate(CATS):
        for i, (m, title) in enumerate((("image_AUROC", "Image AUROC"), ("pixel_AUPRO", "Pixel AUPRO"))):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.set_xscale("log")
            ax.axhline(full.loc[cat, m], color=REF, ls="--", lw=1.2)
            for model, color, label, marker in (("patchcore", BLUE, "PatchCore", "o"), ("winclip", VIOLET, "WinCLIP", "^")):
                x = [n for n in NS if (cat, model, n) in g.index]
                y = [g.loc[(cat, model, n), (m, "mean")] for n in x]
                e = [g.loc[(cat, model, n), (m, "std")] for n in x]
                ax.errorbar(x, y, yerr=e, color=color, marker=marker, lw=1.8, ms=6, capsize=3, label=label)
            z = g.loc[(cat, "winclip", 0), (m, "mean")]
            ax.plot([0.6], [z], marker="^", color=VIOLET, mfc="none", ms=8, ls="none", label="WinCLIP zero-shot")
            ax.set_xticks([0.6] + NS)
            ax.set_xticklabels(["0"] + [str(n) for n in NS])
            ax.minorticks_off()
            if i == 0:
                lo, hi = ax.get_ylim()
                if hi - lo < 0.04:
                    ax.set_ylim(hi - 0.04, hi)
            if i == 0:
                ax.set_title(cat, loc="left", fontsize=11, color=INK)
            if j == 0:
                ax.set_ylabel(title)
            if i == 1:
                ax.set_xlabel("number of good training / reference images N")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.text(0.01, 0.005, "Dashed line: PatchCore with all training images (mean of 3 seeds). Points: mean of 3 seeds, bars: std "
             "(zero-shot: 1 run).", fontsize=8, color=INK2)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(FIG / "fewshot_3cat.png", dpi=150, facecolor=SURFACE)
    print("\n".join(md))


if __name__ == "__main__":
    main()
