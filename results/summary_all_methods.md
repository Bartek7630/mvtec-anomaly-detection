| method | runs | training images | image AUROC | pixel AUROC | training time [s] | model forward [ms/image] |
|---|---|---|---|---|---|---|
| PatchCore | 3 seeds | 320 | 0.960 ± 0.013 | 0.981 ± 0.001 | 37 | 12.6 |
| PaDiM | 3 seeds | 320 | 0.829 ± 0.022 | 0.975 ± 0.001 | 38 | 4.7 |
| STFPM | 3 seeds | 320 | 0.793 ± 0.087 | 0.946 ± 0.032 | 1243 | 10.1 |
| EfficientAD (S) | 3 seeds | 288 | 0.952 ± 0.012 | 0.968 ± 0.006 | 7312 | 12.4 |
| YOLOv8s, 640 px (supervised) | 4 folds | ~120 (~90 defective) | 0.908 ± 0.039 | 0.978 ± 0.008 † | - | - |
| WinCLIP zero-shot | 1 run | 0 | 0.720 | 0.909 | 0 (no training) | - |
| WinCLIP, 10 reference images | 3 draws | 10 | 0.846 ± 0.031 | 0.971 ± 0.005 | 0 (no training) | - |

`±` = standard deviation over seeds (YOLO: over folds). EfficientAD: 3 seeds, 288 training images (10% held out for normalisation). WinCLIP uses 240 x 240 input and needs no training; the 10 reference images are drawn like the PatchCore few-shot images. `-` = not measured. † YOLO pixel AUROC comes from a sample of pixels and a map built from boxes: not comparable with the anomaly maps. Model forward = batch 1, RTX 4060 Laptop, fp32, 256 x 256 input, median of 100 runs, without reading files (`analysis/time_inference.py`); YOLO was not timed.
