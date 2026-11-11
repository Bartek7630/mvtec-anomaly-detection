| method | runs | training images | image AUROC | pixel AUROC | training time [s] | model forward [ms/image] |
|---|---|---|---|---|---|---|
| PatchCore | 3 seeds | 320 | 0.960 ± 0.013 | 0.981 ± 0.001 | 37 | 12.6 |
| PaDiM | 3 seeds | 320 | 0.829 ± 0.022 | 0.975 ± 0.001 | 38 | 4.7 |
| STFPM | 3 seeds | 320 | 0.793 ± 0.087 | - | 139-970 (few-shot runs; full run not saved) | 10.1 |
| EfficientAD (S) | 1 run | 288 | 0.965 | 0.962 | 7598 | 12.4 |
| YOLOv8s, 640 px (supervised) | 4 folds | ~120 (~90 defective) | 0.908 ± 0.039 | 0.978 ± 0.008 † | - | - |

`±` = standard deviation over seeds (YOLO: over folds). EfficientAD: a single run, so no spread; 288 training images (10% held out for normalisation). STFPM: the full-data run's time was not saved; the range comes from its few-shot runs (same 2000 steps). `-` = not measured. † YOLO pixel AUROC comes from a sample of pixels and a map built from boxes: not comparable with the anomaly maps. Model forward = batch 1, RTX 4060 Laptop, fp32, 256 x 256 input, median of 100 runs, without reading files (`analysis/time_inference.py`); YOLO was not timed.
