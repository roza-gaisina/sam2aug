"""
Inpainting interface for the SAM2AUG pipeline.

This module provides a high-level wrapper for image inpainting.
It abstracts away the underlying implementation (e.g., LaMa).

Input:
    - image with missing region (hole)
    - binary mask indicating region to fill

Output:
    - inpainted RGB image

Design:
- Thin wrapper around lama_inpaint.py
- Ensures consistent interface for the pipeline

External dependency:
- Requires LaMa model and configuration (see config.py)
"""

import torch
import cv2

class LamaInpainter:
    """
    Wrapper around LaMa model for image inpainting.

    Loads the model once and reuses it for multiple images.
    """

    def __init__(self, config_path: str, checkpoint_path: str, device: str = None):
        """
        Args:
            config_path: path to LaMa config file
            checkpoint_path: path to LaMa checkpoint directory
            device: "cuda" or "cpu"
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        print(f"[LamaInpainter] Using device: {self.device}")
        from .lama_inpaint import build_lama_model
        self.model = build_lama_model(config_path, checkpoint_path, self.device)
        
    def inpaint(self, image_rgb, mask):
        """
        Inpaint masked regions in an image.

        Args:
            image_rgb: np.ndarray (H, W, 3), RGB image
            mask: np.ndarray (H, W), binary mask

        Returns:
            np.ndarray (H, W, 3), inpainted RGB image
        """
        from .lama_inpaint import inpaint_img_with_lama
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        result_bgr = inpaint_img_with_lama(
            image_bgr,
            mask,
            self.model,
            self.device
        )

        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)