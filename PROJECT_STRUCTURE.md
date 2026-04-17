# 📁 PROJECT_STRUCTURE.md

## Overview

The repository is organized to reflect the full experimental workflow of this thesis, from dataset preparation to augmentation, evaluation, and analysis.

The structure follows three main layers:

1. **Core pipeline (`sam2aug/`)** → object-centric augmentation logic  
2. **Dataset layer (`datasets/`)** → curated inputs and generated outputs  
3. **Experiment layer (`experiments/`)** → dataset generation and evaluation  

This separation ensures modularity, reproducibility, and clarity.

---

## Top-Level Structure

```text
sam2aug/
├── sam2aug/          # Core augmentation pipeline
├── datasets/         # Dataset definitions, inputs, and generated outputs
├── experiments/      # Dataset generation and evaluation scripts
├── notebooks/        # Analysis and visualization
│

├── setup.py          # Package installation
├── pyproject.toml    # Packaging config
├── INSTALL.md          # Environment setup
└── PROJECT_STRUCTURE.md
```

---

# 1. Core Pipeline (`sam2aug/`)

```text
sam2aug/
├── pipeline.py
├── segmenter.py
├── postprocessor.py
├── inpainter.py
├── lama_inpaint.py
├── relocator.py
├── config.py         # Central configuration
└── __init__.py
```

### Purpose

Implements the **object-centric augmentation pipeline** sam2aug.

### Key Components

- **`pipeline.py`**  
  Orchestrates the full pipeline:
  - segmentation  
  - object extraction  
  - inpainting  
  - transformation and relocation  

- **`segmenter.py`**  
  SAM2-based object segmentation using bounding box prompts.

- **`postprocessor.py`**  
  Derives:
  - object cutouts  
  - background with removed object  

- **`inpainter.py` / `lama_inpaint.py`**  
  Background reconstruction using LaMa.

- **`relocator.py`**  
  Object transformation and compositing:
  - scaling  
  - rotation  
  - placement  
  - blending  

---

# 2. Dataset Layer (`datasets/`)

```text
datasets/
├── inputs/
│   ├── imagenet/
│   │   └── meta/
│   ├── donor_images_geirhos_square/
│   └── object_relocation_backgrounds/
│
├── outputs/
│   ├── geirhos_texture_only/
│   ├── geirhos_texture_plus_edges/
│   ├── geirhos_texture_nst_content/
│   ├── geirhos_texture_adain/
│   ├── inpainting_rescale_size_for_prediction/
│   └── object_relocation_background_shift/
│
├── build_selected_manifest.py
├── filter_imagenet_val_images_for_experiments.py
├── *_selected_val_images.txt
└── *_wnid_to_class_mapping.json
```

---

## Purpose

Defines all **inputs and outputs of dataset generation**.

The repository does **not include the full ImageNet dataset**.  
Instead, it stores **curated subsets via manifests and metadata**.

---

## Inputs (`datasets/inputs/`)

### ImageNet metadata
- Manifest files (`*.json`)
- Selected image lists (`*.txt`)
- Class mappings (`*.json`)

Define which images are used in experiments

---

### Donor textures
- `donor_images_geirhos_square/`

Used for:
- shape–texture experiments

---

### Relocation backgrounds
- `openwater.jpg`
- `surface.jpg`

Used for:
- background shift experiment

---

## Outputs (`datasets/outputs/`)

Contains all generated datasets.

Each dataset includes:
- `images/` → generated samples  
- `meta/` → manifests and metadata  

### Dataset families

- **Shape–texture datasets**
  - `geirhos_texture_*`

- **Object rescaling**
  - `inpainting_rescale_size_for_prediction`

- **Object relocation**
  - `object_relocation_background_shift`

---

## Dataset preparation scripts

- **`build_selected_manifest.py`**  
  Creates structured manifests from selected image lists

- **`filter_imagenet_val_images_for_experiments.py`**  
  Filters images where all models predict correctly  
  → ensures clean experimental baseline

---

# 3. Experiments (`experiments/`)

```text
experiments/
├── configs/
├── evaluation_results/
│
├── generate_*.py
└── evaluate_*.py
```

---

## Purpose

Defines the **experimental workflow**:
- dataset generation
- model evaluation
- metric computation

---

## Configurations (`configs/`)

```text
configs/
├── base.yaml
├── geirhos_texture_*.yaml
├── inpainting_rescale_size_for_prediction.yaml
└── object_relocation_background_shift.yaml
```

- Centralized experiment definitions
- Control:
  - pipeline behavior  
  - dataset parameters  
  - output locations  

---

## Generation scripts

```text
generate_geirhos.py
generate_inpainting_rescale_size_for_prediction.py
generate_object_relocation_background_shift.py
```

### Role

- Load configuration
- Iterate over dataset manifests
- Run SAM2AUG
- Store generated datasets

---

## Evaluation scripts

```text
evaluate_geirhos.py
evaluate_inpainting_rescale_size_for_prediction.py
evaluate_object_relocation_background_shift.py
```

### Role

- Run pretrained models
- Compute metrics:
  - accuracy (top-1 / top-5)
  - shape vs. texture decisions
  - robustness measures
- Save results as CSV

---

## Evaluation results

```text
evaluation_results/
├── geirhos_eval_results/
├── inpainting_rescale_size_eval_results/
├── object_relocation_background_shift_eval_results/
└── original_eval_results/
```

- Structured outputs of evaluation scripts
- Used for:
  - notebooks
  - thesis figures

---

# 4. Notebooks (`notebooks/`)

Contains analysis workflows:
- plotting results
- aggregating metrics
- generating thesis figures

---
