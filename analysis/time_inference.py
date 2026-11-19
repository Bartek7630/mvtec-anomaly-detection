"""GPU vs. CPU: how long does one forward pass of each method take on THIS machine?

What is measured
----------------
Only the model forward pass on an image tensor that is already on the device (256 x 256, batch of 1 and of 16):
no reading of image files, no resizing, no metrics, no saving. This is the number to compare with the
"about 2 ms per image" in the EfficientAD paper (RTX A6000). `test_s / 160` from results.csv is much larger because
it contains everything else.

Notes
-----
* EfficientAD and STFPM are timed WITHOUT training: their speed depends only on the architecture, not on the weights.
* PaDiM (WideResNet-50, layers 1+2+3, 200 features, as in run_experiment.py; until 19.12 this script wrongly used ResNet-18)
  and PatchCore need a fitted model (PatchCore: the size of the memory bank matters), so they are fitted here on
  --fit-images training images (default 320 = all) with the same settings as run_experiment.py. PatchCore fitting can take minutes.
* On a GPU every timing is bracketed by torch.cuda.synchronize(), because CUDA calls are asynchronous.
* CPU timing (--device cpu): PaDiM and PatchCore are still FITTED on the GPU if there is one (fitting on a CPU would take very long),
  and the fitted model is then moved to the CPU, so the memory bank has the same size as in the GPU measurement. Only the forward
  pass is timed on the CPU. Defaults for the CPU are lighter: batch 1 only, 5 warm-up passes, 30 timed passes. The processor name,
  the number of logical cores and the number of torch threads are written to the CSV. --threads 1 imitates a weak single-core PC.
  The CPU results go to results/inference_time_cpu.csv (not to inference_time.csv, which is used by make_summary_table.py).

"""

import argparse
import os
import platform
import subprocess
import random
import shutil
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT_REPO = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="MVTec AD category folder (needs train/good and test/good)")
    p.add_argument("--models", nargs="+", default=["efficientad", "stfpm", "padim", "patchcore"],
                   choices=["efficientad", "stfpm", "padim", "patchcore"])
    p.add_argument("--device", default="auto", help="auto, cuda or cpu")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=None, help="default: 1 and 16 on a GPU, 1 on a CPU")
    p.add_argument("--warmup", type=int, default=None, help="untimed forward passes before measuring (default: 20 on a GPU, 5 on a CPU)")
    p.add_argument("--runs", type=int, default=None, help="timed forward passes per batch size (default: 100 on a GPU, 30 on a CPU)")
    p.add_argument("--threads", type=int, default=None, help="CPU: number of torch threads (default: torch's default; 1 = single core)")
    p.add_argument("--fit-images", type=int, default=320, help="PaDiM / PatchCore: number of training images used for fitting")
    p.add_argument("--efficientad-modelsize", default="small", choices=["small", "medium"])
    p.add_argument("--out", default=None, help="default: results/inference_time.csv (GPU) or results/inference_time_cpu.csv (CPU)")
    return p.parse_args()


def cpu_name() -> str:
    """Marketing name of the processor, e.g. 'AMD Ryzen 7 7840HS'. Falls back to platform.processor()."""
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        elif platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out
    except Exception:
        pass
    return platform.processor() or "unknown"


def build_model(name: str, root: Path, n_fit: int, efficientad_size: str, device: torch.device):
    """Return the torch module (model.model) in eval mode on `device`. Fitting uses the GPU whenever there is one."""
    fit_on_gpu = torch.cuda.is_available()
    from anomalib.models import EfficientAd, Padim, Patchcore, Stfpm

    if name == "efficientad":
        model = EfficientAd(model_size=efficientad_size)
    elif name == "stfpm":
        model = Stfpm(backbone="resnet18", layers=("layer1", "layer2", "layer3"))
    else:
        from anomalib.data import Folder
        from anomalib.data.utils import ValSplitMode
        from anomalib.engine import Engine

        model = (Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), pre_trained=True,
                           coreset_sampling_ratio=0.01, num_neighbors=9) if name == "patchcore"
                 else Padim(backbone="wide_resnet50_2", layers=["layer1", "layer2", "layer3"], pre_trained=True, n_features=200))
        files = sorted(f for f in (root / "train" / "good").iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        subset = random.Random(0).sample(files, min(n_fit, len(files)))
        tmp = Path(tempfile.mkdtemp(prefix="timing_fit_"))
        try:
            for f in subset:
                shutil.copy(f, tmp / f.name)
            dm = Folder(name="timing", root=root, normal_dir=tmp, normal_test_dir="test/good", train_batch_size=16,
                        eval_batch_size=16, num_workers=0, val_split_mode=ValSplitMode.SAME_AS_TEST)
            t0 = time.time()
            Engine(accelerator="gpu" if fit_on_gpu else "cpu", default_root_dir=str(tmp / "out"),
                   logger=False, enable_progress_bar=False).fit(model=model, datamodule=dm)
            print(f"  fitted {name} on {len(subset)} images in {time.time() - t0:.0f} s")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return model.model.to(device).eval()


def load_test_batch(root: Path, n: int, device: torch.device) -> torch.Tensor:
    """n real test images, resized to 256 x 256 and normalised, as one tensor on the device."""
    files = sorted(f for f in (root / "test" / "good").iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    files = [files[i % len(files)] for i in range(n)]
    x = torch.stack([torch.from_numpy(__import__("numpy").asarray(Image.open(f).convert("RGB").resize((256, 256)),
                                                                     dtype="float32") / 255.0).permute(2, 0, 1) for f in files])
    return ((x - MEAN) / STD).to(device)


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def time_forward(net, x: torch.Tensor, warmup: int, runs: int, device: torch.device) -> list:
    for _ in range(warmup):
        net(x)
    sync(device)
    times = []
    for _ in range(runs):
        sync(device)
        t0 = time.perf_counter()
        net(x)
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main():
    args = parse_args()
    root = Path(args.root)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    on_gpu = device.type == "cuda"
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    cpu = cpu_name()
    n_threads = torch.get_num_threads()
    if args.batch_sizes is None:
        args.batch_sizes = [1, 16] if on_gpu else [1]
    if args.warmup is None:
        args.warmup = 20 if on_gpu else 5
    if args.runs is None:
        args.runs = 100 if on_gpu else 30
    if args.out is None:
        args.out = str(ROOT_REPO / "results" / ("inference_time.csv" if on_gpu else "inference_time_cpu.csv"))
    print(f"device: {device}  {gpu}  | torch {torch.__version__} | {platform.platform()}")
    print(f"cpu: {cpu} | logical cores: {os.cpu_count()} | torch threads: {n_threads}")
    torch.set_float32_matmul_precision("high")

    x_all = load_test_batch(root, max(args.batch_sizes), device)
    rows = []
    for name in args.models:
        print(f"\n{name}: building model ...")
        net = build_model(name, root, args.fit_images, args.efficientad_modelsize, device)
        for bs in args.batch_sizes:
            times = time_forward(net, x_all[:bs], args.warmup, args.runs, device)
            per_img = sorted(t / bs for t in times)
            row = {
                "date": datetime.now().isoformat(timespec="seconds"), "model": name, "device": device.type, "gpu": gpu,
                "cpu": cpu, "logical_cores": os.cpu_count(), "torch_threads": n_threads, "torch": torch.__version__, "batch_size": bs, "warmup": args.warmup, "runs": args.runs,
                "median_ms_per_image": round(statistics.median(per_img), 3),
                "mean_ms_per_image": round(statistics.fmean(per_img), 3),
                "p95_ms_per_image": round(per_img[int(0.95 * (len(per_img) - 1))], 3),
                "images_per_s": round(1000.0 / statistics.median(per_img), 1),
                "note": "untrained weights (speed does not depend on weights)" if name in ("efficientad", "stfpm")
                        else f"fitted on {args.fit_images} training images",
            }
            rows.append(row)
            print(f"  batch {bs:>2}: median {row['median_ms_per_image']:.2f} ms/image, p95 {row['p95_ms_per_image']:.2f} ms, "
                  f"{row['images_per_s']:.0f} images/s")
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    pd.concat([pd.read_csv(out), new], ignore_index=True).to_csv(out, index=False) if out.exists() else new.to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows to {out}")
    print("\nSummary (batch 1 = one image at a time):")
    print(new[new["batch_size"] == min(args.batch_sizes)][["model", "median_ms_per_image", "p95_ms_per_image", "images_per_s"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
