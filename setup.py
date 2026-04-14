from pathlib import Path
from setuptools import setup, find_packages

PACKAGE_ROOT = Path(__file__).resolve().parent
VERSION_PATH = PACKAGE_ROOT / "sam2aug" / "VERSION"

with open(VERSION_PATH, "r", encoding="utf-8") as f:
    version = f.read().strip()

setup(
    name="sam2aug",
    version=version,
    description=(
        "Segmentation, inpainting, and object relocation pipeline "
        "for controlled image augmentation experiments."
    ),
    author="Roza Gaisina",
    packages=find_packages(),
    python_requires=">=3.10,<3.11",
    install_requires=[
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "opencv-python>=4.10.0,<5",
        "omegaconf>=2.3.0,<3",
        "numpy>=1.26,<2.0",
        "Pillow>=11.0,<12",
        "pandas>=2.3,<3",
        "hydra-core>=1.3.2,<2",
        "pytorch-lightning>=1.2.9,<2",
    ],
)