"""Turn YOLOv8 detections into the same numbers that the anomaly detection methods produce.

For every test image of one fold:
  * image score = the highest confidence among the detected boxes (0 if there are none)
  * anomaly map = zeros with every box filled with its confidence (the maximum where boxes overlap)
Then image-level AUROC and pixel-level AUROC are computed, and the image scores are saved in the same
format as the scores of run_experiment.py --save-scores (columns: path, label, score).
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="trained weights, e.g. runs/detect/fold0_all_640/weights/last.pt")
    p.add_argument("--data-dir", required=True, help="folder created by make_yolo_data.py, e.g. yolo_data/fold0")
    p.add_argument("--root", required=True, help="MVTec AD category folder (needed for the ground-truth masks)")
    p.add_argument("--folds", default="splits/screw_folds.csv")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--imgsz", type=int, required=True, help="use the same size as in training")
    p.add_argument("--name", default=None, help="label of this run in the output files (default: weights folder name)")
    p.add_argument("--scores-dir", default="scores_yolo")
    p.add_argument("--out", default="results_yolo.csv", help="CSV file to append the metrics to")
    p.add_argument("--neg-per-image", type=int, default=25000, help="background pixels sampled per image for pixel AUROC")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def image_score_and_map(result):
    """Score = largest box confidence; map = boxes filled with their confidences."""
    h, w = result.orig_shape
    amap = np.zeros((h, w), dtype=np.float32)
    conf = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    for (x1, y1, x2, y2), c in zip(boxes, conf):
        x1, y1 = max(int(round(x1)), 0), max(int(round(y1)), 0)
        x2, y2 = min(int(round(x2)), w), min(int(round(y2)), h)
        amap[y1:y2, x1:x2] = np.maximum(amap[y1:y2, x1:x2], c)
    score = float(conf.max()) if len(conf) > 0 else 0.0
    return score, amap


def load_mask(root: Path, defect_type: str, stem: str, shape):
    """Ground-truth mask as a boolean array (all False for good images)."""
    if defect_type == "good":
        return np.zeros(shape, dtype=bool)
    mask = cv2.imread(str(root / "ground_truth" / defect_type / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != tuple(shape):
        raise ValueError(f"Missing or wrongly sized mask for {defect_type}/{stem}")
    return mask > 0


def main():
    args = parse_args()
    root, data_dir = Path(args.root), Path(args.data_dir)
    rng = np.random.default_rng(args.seed)
    name = args.name or Path(args.weights).parent.parent.name

    folds = pd.read_csv(args.folds)
    test_rows = folds[folds["fold"] == args.fold]
    lookup = {f"{r.defect_type}_{Path(r.path).stem}": r for r in test_rows.itertuples()}

    model = YOLO(args.weights)
    results = model.predict(str(data_dir / "images" / "test"), conf=0.001, imgsz=args.imgsz, stream=True, verbose=False)

    rows, pos_values, neg_values = [], [], []
    for result in results:
        key = Path(result.path).stem
        if key not in lookup:
            raise SystemExit(f"Image {key} is not in fold {args.fold} of {args.folds}")
        row = lookup[key]
        score, amap = image_score_and_map(result)
        assert abs(float(amap.max()) - score) < 1e-6, f"map and score disagree for {key}"
        mask = load_mask(root, row.defect_type, Path(row.path).stem, amap.shape)

        pos_values.append(amap[mask])
        background = amap[~mask]
        neg_values.append(rng.choice(background, size=min(args.neg_per_image, background.size), replace=False))
        rows.append((row.path, int(row.label), score))

    if len(rows) != len(test_rows):
        raise SystemExit(f"Predicted {len(rows)} images but fold {args.fold} has {len(test_rows)}")

    labels = [r[1] for r in rows]
    scores = [r[2] for r in rows]
    image_auroc = roc_auc_score(labels, scores)

    pos, neg = np.concatenate(pos_values), np.concatenate(neg_values)
    pixel_auroc = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])

    scores_file = Path(args.scores_dir) / f"{name}_fold{args.fold}.csv"
    scores_file.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score"])
        for path, label, score in sorted(rows):
            w.writerow([path, label, round(score, 6)])

    out = Path(args.out)
    new_file = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["run", "imgsz", "fold", "n_images", "image_AUROC", "pixel_AUROC"])
        w.writerow([name, args.imgsz, args.fold, len(rows), round(image_auroc, 4), round(pixel_auroc, 4)])

    print(f"{name} | fold {args.fold} | {len(rows)} images")
    print(f"image AUROC: {image_auroc:.4f}")
    print(f"pixel AUROC: {pixel_auroc:.4f}  ({len(pos)} defect pixels, {len(neg)} sampled background pixels)")
    print(f"Saved {scores_file} and appended to {out}")


if __name__ == "__main__":
    main()