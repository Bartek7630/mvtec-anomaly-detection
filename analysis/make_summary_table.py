"""Day 21.11: one table for all methods on MVTec AD 'screw' (full training data).

Input : results/four_categories.csv   PatchCore, PaDiM (screw rows, seeds 1, 2, 42): AUROC, fit_s
        results/stfpm_screw.csv       STFPM (seeds 1, 2, 42): image AUROC; results/stfpm_fewshot.csv for pixel AUROC and fit_s
        results/efficientad_screw.csv EfficientAD, ONE run (seed 42), 288 training images
        results/yolo_folds.csv        YOLOv8s 640 px, 4 fold models
        results/inference_time.csv    forward pass per image, batch 1, median (analysis/time_inference.py)
Output: results/summary_all_methods.csv and results/summary_all_methods.md

What is NOT in the table, on purpose
* STFPM: pixel AUROC for the full-data run was not saved, so it is left empty.
* STFPM: fit time of the full run was not saved. Few-shot runs have the same 2000 steps but take 150-940 s depending on N
  (more validation passes when the epochs are shorter), so no single number is honest; the table shows that range.
* YOLO: no training time and no latency were measured, and its pixel AUROC is computed from boxes on a sample of pixels, so it is not
  comparable with the anomaly maps (marked with a dagger).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def ms(x, std=None, nd=3):
    return f"{x:.{nd}f}" if std is None or pd.isna(std) else f"{x:.{nd}f} ± {std:.{nd}f}"


def main():
    four = pd.read_csv(RES / "four_categories.csv")
    four = four[four["category"] == "screw"]
    lat = pd.read_csv(RES / "inference_time.csv")
    lat = lat[lat["batch_size"] == 1].set_index("model")["median_ms_per_image"]
    rows = []

    for key, name in (("patchcore", "PatchCore"), ("padim", "PaDiM")):
        d = four[four["model"] == key]
        assert len(d) == 3, (key, len(d))
        rows.append(dict(method=name, runs="3 seeds", train_images=320,
                         image_AUROC=ms(d.image_AUROC.mean(), d.image_AUROC.std()),
                         pixel_AUROC=ms(d.pixel_AUROC.mean(), d.pixel_AUROC.std()),
                         train_time_s=f"{d.fit_s.mean():.0f}", ms_per_image=f"{lat[key]:.1f}"))

    st = pd.read_csv(RES / "stfpm_screw.csv")
    st = st[st["epochs"] == 100]
    few = pd.read_csv(RES / "stfpm_fewshot.csv")
    assert len(st) == 3
    rows.append(dict(method="STFPM", runs="3 seeds", train_images=320,
                     image_AUROC=ms(st.image_AUROC.mean(), st.image_AUROC.std()), pixel_AUROC="-",
                     train_time_s=f"{few.fit_s.min():.0f}-{few.fit_s.max():.0f} (few-shot runs; full run not saved)", ms_per_image=f"{lat['stfpm']:.1f}"))

    ea = pd.read_csv(RES / "efficientad_screw.csv")
    ea = ea[ea["n_train"] > 100]
    assert len(ea) == 1
    e = ea.iloc[0]
    rows.append(dict(method="EfficientAD (S)", runs="1 run", train_images=int(e.n_train),
                     image_AUROC=ms(e.image_AUROC), pixel_AUROC=ms(e.pixel_AUROC),
                     train_time_s=f"{e.fit_s:.0f}", ms_per_image=f"{lat['efficientad']:.1f}"))

    yo = pd.read_csv(RES / "yolo_folds.csv")
    yo = yo[yo["imgsz"] == 640]
    assert len(yo) == 4
    rows.append(dict(method="YOLOv8s, 640 px (supervised)", runs="4 folds", train_images="~120 (~90 defective)",
                     image_AUROC=ms(yo.image_AUROC.mean(), yo.image_AUROC.std()),
                     pixel_AUROC=ms(yo.pixel_AUROC.mean(), yo.pixel_AUROC.std()) + " †",
                     train_time_s="-", ms_per_image="-"))

    out = pd.DataFrame(rows)
    out.to_csv(RES / "summary_all_methods.csv", index=False)
    header = ["method", "runs", "training images", "image AUROC", "pixel AUROC", "training time [s]", "model forward [ms/image]"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in out.itertuples(index=False)]
    notes = ("\n\n`±` = standard deviation over seeds (YOLO: over folds). EfficientAD: a single run, so no spread; 288 training images "
             "(10% held out for normalisation). STFPM: the full-data run's time was not saved; the range comes from its few-shot runs (same 2000 steps). `-` = not measured. "
             "† YOLO pixel AUROC comes from a sample of pixels and a map built from boxes: not comparable with the anomaly maps. "
             "Model forward = batch 1, RTX 4060 Laptop, fp32, 256 x 256 input, median of 100 runs, without reading files "
             "(`analysis/time_inference.py`); YOLO was not timed.\n")
    (RES / "summary_all_methods.md").write_text("\n".join(lines) + notes, encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
