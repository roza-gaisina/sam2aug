"""
Central configuration for the SAM2AUG pipeline.

This file defines paths and global settings required to run the pipeline.

Key principles:
- All paths are configurable
- No hardcoded machine-specific dependencies
- External models are referenced via environment variables

External dependencies:
- SAM2 repository (segmentation)
- LaMa repository (inpainting)

Users must set:
    SAM2_HOME
    LAMA_HOME

before running the pipeline.

This file does NOT contain experiment-specific parameters.
Those belong to experiment configs (e.g., YAML files).
"""

import os
import torch

def _get_env_var(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise ValueError(
            f"Environment variable '{name}' is not set. "
            f"See SETUP.md for instructions."
        )
    return value

# =========================
# Output / workspace for intermediate results of SAM2AUG
# =========================

#PIPELINE_OUTPUT_DIR = _get_env_var("SAM2AUG_OUTPUT_DIR")

# =========================
# External dependencies
# =========================

SAM2_HOME_DIR = _get_env_var("SAM2_HOME")
LAMA_HOME_DIR = _get_env_var("LAMA_HOME")

# =========================
# Device
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# SAM2 configuration
# =========================

SAM2_CHECKPOINT = os.path.join(
    SAM2_HOME_DIR, "checkpoints/sam2.1_hiera_base_plus.pt"
)

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# =========================
# Dataset configuration (COCO)
# =========================

# DATASET_PATH = "datasets/coco"
# IMAGE_DIR = os.path.join(DATASET_PATH, "train2017")
# ANNOTATION_FILE = os.path.join(
#     DATASET_PATH, "annotations/instances_train2017.json"
# )

# =========================
# Segmentation outputs
# =========================

# SEGMENTATION_DIR = os.path.join(PIPELINE_OUTPUT_DIR, "segmentation")

# SEGMENTED_OBJECTS_DIR = os.path.join(SEGMENTATION_DIR, "objects")
# SEGMENTED_BACKGROUNDS_DIR = os.path.join(SEGMENTATION_DIR, "backgrounds")
# SEGMENTATION_MASKS_DIR = os.path.join(SEGMENTATION_DIR, "masks")

# =========================
# LaMa configuration
# =========================

LAMA_CONFIG_PATH = os.path.join(
    LAMA_HOME_DIR, "configs/prediction/default.yaml"
)

LAMA_CHECKPOINT_PATH = os.path.join(
    LAMA_HOME_DIR, "big-lama"
)