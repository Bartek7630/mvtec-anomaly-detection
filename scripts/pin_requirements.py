"""Writes requirements.txt with the versions installed in THIS environment.

Run it inside the venv in which all the experiments were run:
    python scripts/pin_requirements.py
It pins the packages the project imports directly, plus open-clip-torch (needed by WinCLIP) and prints
the torch / torchvision versions, which are installed separately (CUDA build) and are only noted in a comment.
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ["anomalib", "lightning", "pandas", "numpy", "matplotlib", "pillow", "scikit-learn",
            "ultralytics", "open-clip-torch", "timm", "openpyxl"]
lines = []
for name in PACKAGES:
    try:
        lines.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        print(f"not installed, skipped: {name}")
tv = {}
for name in ("torch", "torchvision"):
    try:
        tv[name] = version(name)
    except PackageNotFoundError:
        tv[name] = "?"
text = "\n".join(lines) + (f"\n# torch {tv['torch']} / torchvision {tv['torchvision']} are installed separately "
                           "(CUDA build), see README\n")
Path(__file__).resolve().parents[1].joinpath("requirements.txt").write_text(text, encoding="utf-8")
print(text)
