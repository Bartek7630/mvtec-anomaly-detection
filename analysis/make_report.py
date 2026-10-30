"""Build summary tables (Markdown) and figures from the CSV files in results/."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

METRICS = ["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_AUPRO", "pixel_F1Score"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, REF = "#2a78d6", "#d9822b", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def mean_std(df, by):
    g = df.groupby(by)[METRICS].agg(["mean", "std"])
    return g


def fmt(g, m, idx):
    return f"{g.loc[idx, (m, 'mean')]:.3f} ± {g.loc[idx, (m, 'std')]:.3f}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def four_categories():
    df = pd.read_csv(RES / "four_categories.csv")
    g = mean_std(df, ["category", "model"])
    order = ["screw", "bottle", "transistor", "carpet"]
    rows = []
    for cat in order:
        for model in ("patchcore", "padim"):
            rows.append([cat, {"patchcore": "PatchCore", "padim": "PaDiM"}[model]] +
                        [fmt(g, m, (cat, model)) for m in METRICS])
    table = md_table(["category", "method", "image AUROC", "image F1", "pixel AUROC", "pixel AUPRO", "pixel F1"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), facecolor=SURFACE)
    for ax, m, title in zip(axes, ["image_AUROC", "pixel_AUPRO"], ["Image AUROC", "Pixel AUPRO"]):
        style(ax)
        x = np.arange(len(order))
        for k, (model, color, label) in enumerate([("patchcore", BLUE, "PatchCore"), ("padim", ORANGE, "PaDiM")]):
            mu = [g.loc[(c, model), (m, "mean")] for c in order]
            sd = [g.loc[(c, model), (m, "std")] for c in order]
            ax.bar(x + (k - 0.5) * 0.38, mu, 0.34, yerr=sd, color=color, label=label, capsize=3,
                   error_kw={"lw": 1.2, "ecolor": INK2})
            for xi, v, e in zip(x + (k - 0.5) * 0.38, mu, sd):
                ax.text(xi, v + e + 0.008, f"{v:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_ylim(0.7, 1.04)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(0.98, 0.99))
    fig.suptitle("PatchCore vs PaDiM on four MVTec AD categories (mean ± std, 3 seeds)", x=0.05, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "four_categories.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return table


def few_shot():
    df = pd.read_csv(RES / "few_shot.csv")
    main = df[((df.n_train < 320) & (df.coreset_ratio == 1.0)) | (df.n_train == 320)]
    g = mean_std(main, "n_train")
    ns = [1, 5, 10, 50, 320]
    bank = {1: 1024, 5: 5120, 10: 10240, 50: 51200, 320: 3276}
    rows = [[n, "1%" if n == 320 else "100%", f"{bank[n]:,}"] + [fmt(g, m, n) for m in METRICS] for n in ns]
    table = md_table(["N train images", "coreset", "bank size (patches)", "image AUROC", "image F1",
                      "pixel AUROC", "pixel AUPRO", "pixel F1"], rows)

    panels = [("image_AUROC", "Image AUROC (is the image defective?)", (0.4, 1.04), (0.5, "chance = 0.5")),
              ("image_F1Score", "Image F1 (misleading: 74% of test images are defective)", (0.8, 1.0),
               (0.853, "predict-all-defective = 0.853")),
              ("pixel_AUROC", "Pixel AUROC (localisation)", (0.88, 1.0), None),
              ("pixel_AUPRO", "Pixel AUPRO (localisation, per defect)", (0.6, 1.0), None)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), facecolor=SURFACE)
    for ax, (m, title, ylim, ref) in zip(axes.flat, panels):
        style(ax)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.minorticks_off()
        ax.set_ylim(*ylim)
        ax.set_xlim(0.7, 480)
        if ref:
            ax.axhline(ref[0], color=REF, lw=1, ls=(0, (4, 3)))
            ax.text(470, ref[0] - (ylim[1] - ylim[0]) * 0.012, ref[1], fontsize=8, color=INK2, va="top", ha="right")
        for n in ns:
            v = main[main.n_train == n][m].values
            ax.scatter(np.full(len(v), n) * np.array([0.93, 1, 1.07][:len(v)]), v, s=14, color=BLUE,
                       alpha=0.45, zorder=3, linewidths=0)
        mu = np.array([g.loc[n, (m, "mean")] for n in ns])
        sd = np.array([g.loc[n, (m, "std")] for n in ns])
        ax.plot(ns[:4], mu[:4], color=BLUE, lw=2, zorder=4)
        ax.plot(ns[3:], mu[3:], color=BLUE, lw=2, ls=(0, (3, 2)), zorder=4)
        ax.errorbar(ns[:4], mu[:4], yerr=sd[:4], fmt="o", color=BLUE, ms=8, capsize=3, lw=1.5, zorder=5,
                    mec=SURFACE, mew=1.5)
        ax.errorbar([320], [mu[4]], yerr=[sd[4]], fmt="o", color=BLUE, mfc=SURFACE, ms=8, capsize=3, lw=1.5,
                    zorder=5, mew=2)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
        ax.set_xlabel("number of training images N (log scale)", fontsize=8.5)
        if m == "image_AUROC":
            for n, v in zip(ns, mu):
                off, ha = ((11, -12), "left") if n < 50 else (((-13, -12), "right") if n == 50 else ((0, 11), "center"))
                ax.annotate(f"{v:.3f}", (n, v), textcoords="offset points", xytext=off, ha=ha, fontsize=8.5, color=INK)
    fig.suptitle("PatchCore (WideResNet-50, 256 px) on MVTec AD 'screw': effect of the number of training images",
                 fontsize=12.5, color=INK, x=0.06, ha="left")
    fig.text(0.06, 0.012, "Large dots: mean ± std over 3 random draws; small dots: single draws. N = 1-50 use a 100% coreset; "
             "N = 320 (hollow, dashed) uses a 1% coreset.", fontsize=8, color=INK2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(FIG / "few_shot_curve.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    ctrl = df[df.n_train == 50]
    gc = mean_std(ctrl, "coreset_ratio")
    ratios = [1.0, 0.1, 0.01]
    rows = [[f"{r:.0%}", f"{int(r * 51200):,}"] + [fmt(gc, m, r) for m in METRICS] for r in ratios]
    table_ctrl = md_table(["coreset", "bank size (patches)", "image AUROC", "image F1", "pixel AUROC",
                           "pixel AUPRO", "pixel F1"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), facecolor=SURFACE)
    base_vals = {"image_AUROC": (g.loc[320, ("image_AUROC", "mean")], (0.7, 1.0)),
                 "pixel_AUPRO": (g.loc[320, ("pixel_AUPRO", "mean")], (0.7, 0.96))}
    for ax, m, t in zip(axes, ["image_AUROC", "pixel_AUPRO"], ["Image AUROC", "Pixel AUPRO"]):
        style(ax)
        xs = [0, 1, 2]
        for x, r in zip(xs, ratios):
            v = ctrl[ctrl.coreset_ratio == r][m].values
            ax.scatter(np.full(len(v), x) + np.array([-0.06, 0, 0.06])[:len(v)], v, s=14, color=BLUE, alpha=0.45,
                       zorder=3, linewidths=0)
        mu = [gc.loc[r, (m, "mean")] for r in ratios]
        sd = [gc.loc[r, (m, "std")] for r in ratios]
        ax.plot(xs, mu, color=BLUE, lw=2)
        ax.errorbar(xs, mu, yerr=sd, fmt="o", color=BLUE, ms=8, capsize=3, mec=SURFACE, mew=1.5, zorder=5)
        for x, v in zip(xs, mu):
            ax.annotate(f"{v:.3f}", (x, v), xytext=(10, -12), textcoords="offset points", fontsize=9, color=INK)
        bv, ylim = base_vals[m]
        ax.axhline(bv, color=REF, ls=(0, (4, 3)), lw=1)
        ax.text(2.4, bv - 0.005, f"N = 320 (1% coreset): {bv:.2f}", ha="right", va="top", fontsize=8.5, color=INK2)
        ax.set_ylim(*ylim)
        ax.set_xticks(xs)
        ax.set_xticklabels(["100%\n(51,200 patches)", "10%\n(5,120)", "1%\n(512)"])
        ax.set_xlim(-0.3, 2.4)
        ax.set_title(t, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("coreset size at N = 50", fontsize=8.5)
    fig.suptitle("Control: same 50 images, smaller coreset ('screw', PatchCore)", fontsize=12, color=INK, x=0.05,
                 ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "coreset_control.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return table, table_ctrl


def ablation():
    df = pd.read_csv(RES / "ablation_screw.csv")
    df["config"] = (df.model.map({"patchcore": "PatchCore", "padim": "PaDiM"}) + " " + df.backbone + " " +
                    df.image_size.astype(str) + " px" +
                    np.where(df.model == "patchcore", ", k=" + df.num_neighbors.astype("Int64").astype(str), ""))
    g = df.groupby("config", sort=False)[["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score", "fit_s"]].agg(
        ["mean", "std"])
    rows = [[c, f"{g.loc[c, ('image_AUROC', 'mean')]:.3f} ± {g.loc[c, ('image_AUROC', 'std')]:.3f}",
             f"{g.loc[c, ('pixel_AUROC', 'mean')]:.3f}", f"{g.loc[c, ('pixel_F1Score', 'mean')]:.3f}",
             f"{g.loc[c, ('fit_s', 'mean')]:.0f}"] for c in g.index]
    return md_table(["configuration (screw, 3 seeds)", "image AUROC", "pixel AUROC", "pixel F1", "fit time [s]"], rows)


if __name__ == "__main__":
    t_four = four_categories()
    t_few, t_ctrl = few_shot()
    t_abl = ablation()
    out = ("## Four categories\n\n" + t_four + "\n\n## Few-shot (screw)\n\n" + t_few +
           "\n\n## Coreset control (N = 50)\n\n" + t_ctrl + "\n\n## Ablation (screw)\n\n" + t_abl + "\n")
    (RES / "summary.md").write_text(out, encoding="utf-8")
    print(out)
