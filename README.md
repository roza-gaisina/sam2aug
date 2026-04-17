# sam2aug

`sam2aug` is a modular object-centric image augmentation framework developed for controlled robustness experiments on image classification models.

The pipeline combines:

- **segmentation** with **SAM2**
- **background reconstruction** with **LaMa**
- **object transformation and relocation**

It was evaluated during the experiments accompanying master thesis, with a focus on three controlled transformations:

- **shape–texture manipulations**
- **object relocation with background shift**
- **object rescaling on reconstructed backgrounds**

---

## Read this first

Before installing or running anything, read these files in the following order:

1. **`INSTALL.md`**  
   Environment setup, external dependencies, editable installs, required environment variables, and troubleshooting.

2. **`PROJECT_STRUCTURE.md`**  
   Detailed repository structure and the relationship between pipeline code, datasets, experiments, and outputs.

3. **`sam2aug/config.py`**  
   Central configuration for paths, model access, and shared settings.

If you only want the shortest path to a working installation, start with **`SETUP.md`**.

---

## Repository overview

```text
sam2aug/
├── sam2aug/          # Core augmentation pipeline
├── datasets/         # Curated inputs, manifests, donor images, generated outputs
├── experiments/      # Dataset generation and evaluation scripts
├── notebooks/        # Analysis and plotting notebooks
│
├── INSTALL.md
├── PROJECT_STRUCTURE.md
├── setup.py
└── pyproject.toml
```

### Core package

The core implementation lives in `sam2aug/`:

- `pipeline.py` — orchestrates segmentation, extraction, inpainting, augmentation, and relocation
- `segmenter.py` — SAM2-based object segmentation
- `postprocessor.py` — object extraction and source-with-hole creation
- `inpainter.py` / `lama_inpaint.py` — LaMa-based background reconstruction
- `relocator.py` — object transformation, scaling, placement, and blending

## Installation summary

The detailed version is in `INSTALL.md`. The short version is:

### 1. Python and environment

The codebase was developed and tested with:

- Python 3.10.19
- PyTorch 2.9.0 (CUDA 13.0, cu130)
- torchvision 0.24.0 (cu130)
- NVIDIA CUDA runtime 13.0
### 2. Install `sam2aug`

Run this:

```bash
git clone https://github.com/roza-gaisina/sam2aug.git 
cd sam2aug
pip install -e .
```
---

## Example outputs

Below are representative examples created with sam2aug and used in the thesis.  

### Intermediate pipeline outputs

This figure illustrates the main intermediate outputs of the pipeline for a single example:

- original image with bounding box
- predicted segmentation mask
- extracted object
- image with removed object
- inpainted background
- final composited image

![Intermediate outputs of the sam2aug pipeline](docs/images/sam2aug_steps.png?raw=true)

### Shape–texture dataset variants

Representative examples of the shape–texture dataset variants:

- original image
- Texture Only
- Texture + Edges
- Texture NST
- Texture AdaIN
  
![Shape–texture dataset variants](docs/images/shape_texture.png?raw=true)

### Object relocation with background shift

Examples from the relocation dataset with novel backgrounds.

![Object relocation with background shift](docs/images/background_shift.png?raw=true)

### Object rescaling

Examples from the object rescaling dataset.

![Object rescaling on reconstructed background](docs/images/object_rescale.png?raw=true)
