"""Build summary tables and figures from the CSV files in results/."""

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


def fifteen_categories():
    """All 15 MVTec AD categories: PatchCore vs PaDiM, 3 seeds. Needs results_15.csv (repo root or results/)."""
    path = next((p for p in (RES / "results_15.csv", ROOT / "results_15.csv") if p.exists()), None)
    if path is None:
        return None
    df = pd.read_csv(path)
    names = {"patchcore": "PatchCore", "padim": "PaDiM"}
    g = mean_std(df, ["category", "model"])
    order = list(g.xs("patchcore", level="model")[("image_AUROC", "mean")].sort_values().index)  # hardest first

    # mean over the 15 categories, one value per seed; std = spread of that mean across the 3 seeds
    per_seed = df.groupby(["model", "seed"])[METRICS].mean()
    def avg(model, m):
        v = per_seed.loc[model, m]
        return f"{v.mean():.3f} ± {v.std():.3f}"

    rows = []
    for cat in order:
        for model in ("patchcore", "padim"):
            rows.append([cat, names[model]] + [fmt(g, m, (cat, model)) for m in METRICS])
    for model in ("patchcore", "padim"):
        rows.append(["**mean of 15**", names[model]] + [avg(model, m) for m in METRICS])
    table = md_table(["category", "method", "image AUROC", "image F1", "pixel AUROC", "pixel AUPRO", "pixel F1"], rows)

    flat = g.copy()
    flat.columns = [f"{a}_{b}" for a, b in flat.columns]
    flat.reset_index().to_csv(RES / "fifteen_categories_table.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4), facecolor=SURFACE, sharey=True)
    y = np.arange(len(order) + 1)                       # last row = mean of 15
    labels = order + ["mean of 15"]
    for ax, m, title, lo in zip(axes, ["image_AUROC", "pixel_AUPRO"], ["Image AUROC", "Pixel AUPRO"], (0.78, 0.78)):
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
        for k, (model, color) in enumerate([("patchcore", BLUE), ("padim", ORANGE)]):
            mu = [g.loc[(c, model), (m, "mean")] for c in order] + [per_seed.loc[model, m].mean()]
            sd = [g.loc[(c, model), (m, "std")] for c in order] + [per_seed.loc[model, m].std()]
            ax.barh(y + (k - 0.5) * 0.38, mu, 0.34, xerr=sd, color=color, capsize=2,
                    label=names[model], error_kw={"lw": 1.0, "ecolor": INK2})
            for yi, v, e in zip(y + (k - 0.5) * 0.38, mu, sd):
                inside = v > 0.995
                ax.text(v - 0.004 if inside else v + e + 0.004, yi, f"{v:.2f}", va="center",
                        ha="right" if inside else "left", fontsize=7, color="white" if inside else INK)
        ax.axhline(len(order) - 0.5, color=GRID, lw=1.2)
        ax.set_xlim(lo, 1.03)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(0.98, 0.93))
    fig.suptitle("PatchCore vs PaDiM on all 15 MVTec AD categories\n(mean ± std over 3 seeds; hardest for PatchCore at the top)",
                 x=0.03, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(FIG / "fifteen_categories.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return table


def winclip_15():
    """All 15 categories: PatchCore / PaDiM (full train set, 3 seeds) vs WinCLIP (0 and 10 reference images).
    Needs results_winclip_15.csv and results_15.csv (repo root or results/)."""
    p_w = next((p for p in (RES / "results_winclip_15.csv", ROOT / "results_winclip_15.csv") if p.exists()), None)
    p_b = next((p for p in (RES / "results_15.csv", ROOT / "results_15.csv") if p.exists()), None)
    if p_w is None or p_b is None:
        return None
    w, b = pd.read_csv(p_w), pd.read_csv(p_b)
    series = {  # label -> (frame, color)
        "PatchCore (all train images)": b[b.model == "patchcore"],
        "PaDiM (all train images)": b[b.model == "padim"],
        "WinCLIP, 0 images": w[w.n_train == 0],
        "WinCLIP, 10 images": w[w.n_train == 10],
    }
    cols = [BLUE, ORANGE, "#8e8e93", "#7a4fd1"]
    stats = {k: v.groupby("category")[METRICS].agg(["mean", "std"]) for k, v in series.items()}
    order = list(stats["PatchCore (all train images)"][("image_AUROC", "mean")].sort_values().index)

    def cell(k, m, c):
        mu, sd = stats[k].loc[c, (m, "mean")], stats[k].loc[c, (m, "std")]
        return f"{mu:.3f}" + (f" ± {sd:.3f}" if pd.notna(sd) and sd > 0 else "")

    def mean15(k, m):
        v = series[k].groupby(["seed", "category"])[m].mean().groupby("seed").mean() if k != "WinCLIP, 0 images" \
            else series[k].groupby("category")[m].mean().mean()
        return f"{v.mean():.3f}" if hasattr(v, "mean") else f"{v:.3f}"

    rows = [[c] + [cell(k, "image_AUROC", c) for k in series] for c in order]
    rows.append(["**mean of 15**"] + [mean15(k, "image_AUROC") for k in series])
    t_img = md_table(["category"] + list(series), rows)
    rows = [[c] + [cell(k, "pixel_AUPRO", c) for k in series] for c in order]
    rows.append(["**mean of 15**"] + [mean15(k, "pixel_AUPRO") for k in series])
    t_pro = md_table(["category"] + list(series), rows)
    flat = pd.concat({k: v for k, v in stats.items()}, axis=1)
    flat.columns = [f"{a} | {b_}_{c}" for a, b_, c in flat.columns]
    flat.reset_index().to_csv(RES / "winclip_15_table.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.2), facecolor=SURFACE, sharey=True)
    y = np.arange(len(order) + 1)
    for ax, m, title, lo in zip(axes, ["image_AUROC", "pixel_AUPRO"], ["Image AUROC", "Pixel AUPRO"], (0.45, 0.3)):
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
        for k, (name, color) in enumerate(zip(series, cols)):
            mu = [stats[name].loc[c, (m, "mean")] for c in order]
            mu.append(float(np.mean(mu)))
            ax.barh(y + (k - 1.5) * 0.2, mu, 0.18, color=color, label=name)
        ax.axhline(len(order) - 0.5, color=GRID, lw=1.2)
        ax.set_xlim(lo, 1.02)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order + ["mean of 15"])
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", ncol=2, fontsize=8.5, bbox_to_anchor=(0.99, 0.95))
    fig.suptitle("WinCLIP (0 and 10 images) vs PatchCore / PaDiM (full train set) on all 15 categories\n"
                 "(PatchCore, PaDiM, WinCLIP 10: mean over 3 seeds / draws; hardest for PatchCore at the top)",
                 x=0.03, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(FIG / "winclip_15.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return "### Image AUROC\n\n" + t_img + "\n\n### Pixel AUPRO\n\n" + t_pro


if __name__ == "__main__":
    t_four = four_categories()
    t_few, t_ctrl = few_shot()
    t_abl = ablation()
    t_15 = fifteen_categories()
    out = ("## Four categories\n\n" + t_four + "\n\n## Few-shot (screw)\n\n" + t_few +
           "\n\n## Coreset control (N = 50)\n\n" + t_ctrl + "\n\n## Ablation (screw)\n\n" + t_abl + "\n")
    if t_15:
        out += "\n## All 15 categories\n\n" + t_15 + "\n"
    t_wc = winclip_15()
    if t_wc:
        out += "\n## WinCLIP on all 15 categories\n\n" + t_wc + "\n"
    (RES / "summary.md").write_text(out, encoding="utf-8")
    print(out)