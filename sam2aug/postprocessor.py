"""
Postprocessing utilities for object extraction.

This module converts segmentation masks into usable intermediate
representations for the pipeline:

    - extracted object (RGB cutout)
    - source image with object removed (hole image)

Responsibilities:
- apply mask to extract object
- generate masked-out background image
- optionally compute tight crops (if implemented)

It acts as a bridge between segmentation and inpainting/relocation.
"""

import os
import numpy as np
import cv2
from typing import Tuple


def extract_object_and_background(
    image_rgb: np.ndarray,
    mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract object and background from an image using a binary mask.

    Args:
        image_rgb: np.ndarray (H, W, 3), RGB image
        mask: np.ndarray (H, W), binary mask (0/1 or 0/255)

    Returns:
        object_image: image with only object (background = black)
        background_image: image with object removed (object region = black)
    """
    mask_bool = mask.astype(bool)

    # Object image (black background)
    object_image = np.zeros_like(image_rgb)
    object_image[mask_bool] = image_rgb[mask_bool]

    # Background image (object removed)
    background_image = image_rgb.copy()
    background_image[mask_bool] = 0

    return object_image, background_image


def save_segmentation_outputs(
    category: str,
    image_rgb: np.ndarray,
    mask: np.ndarray,
    image_id: str,
    object_index: int,
    object_dir: str,
    background_dir: str,
    mask_dir: str
) -> None:
    """
    Save object, background, and mask images for a segmentation result.

    Args:
        category: class/category name
        image_rgb: original RGB image
        mask: binary mask (0/1 or 0/255)
        image_id: identifier of the source image
        object_index: index of object within image
        object_dir: directory for object images
        background_dir: directory for background images
        mask_dir: directory for masks
    """
    object_img, background_img = extract_object_and_background(image_rgb, mask)

    # Ensure category directories exist
    os.makedirs(os.path.join(object_dir, category), exist_ok=True)
    os.makedirs(os.path.join(background_dir, category), exist_ok=True)
    os.makedirs(os.path.join(mask_dir, category), exist_ok=True)

    # File paths
    filename = f"{image_id}_{object_index}.png"

    object_path = os.path.join(object_dir, category, filename)
    background_path = os.path.join(background_dir, category, filename)
    mask_path = os.path.join(mask_dir, category, filename)

    # Convert mask to 0–255 format
    binary_mask = (mask.astype(np.uint8)) * 255

    # Save images (convert RGB → BGR for OpenCV)
    cv2.imwrite(object_path, cv2.cvtColor(object_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(background_path, cv2.cvtColor(background_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, binary_mask)

    print(
        f"Saved object: {object_path}, "
        f"background: {background_path}, "
        f"mask: {mask_path}"
    )