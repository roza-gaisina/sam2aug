"""
Segmentation module for the SAM2AUG pipeline.

This module wraps SAM2-based segmentation and provides a consistent interface
for extracting object masks from images given bounding boxes.

Input:
    - RGB image (HxWx3)
    - bounding boxes (format depends on dataset)

Output:
    - binary masks for each object

Notes:
- This module is responsible ONLY for segmentation.
- It does not perform any postprocessing or filtering.
- The output format is normalized for downstream pipeline compatibility.

External dependency:
- Requires SAM2 repository and checkpoints (configured via config.py).
"""

import torch
from typing import List, Tuple

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Segmenter:
    """
    Wrapper around SAM2 for image segmentation using bounding boxes.

    This class initializes a SAM2 model and provides a simple interface
    to generate segmentation masks for given bounding boxes.
    """
    def __init__(self, model_config, checkpoint_path, device):
        """
        Args:
            model_config: SAM2 config file path (e.g. "configs/sam2.1/...yaml")
            checkpoint_path: Absolute path to SAM2 checkpoint
            device: "cuda" or "cpu"
        """
        self.device = device

        # Reset Hydra state (required when re-initializing in same process)
        GlobalHydra.instance().clear()

        # Initialize SAM2 model using Hydra config system
        with initialize_config_module(config_module="sam2"):
            self.model = build_sam2(
                config_file=model_config,
                ckpt_path=checkpoint_path,
                device=device,
                apply_postprocessing=False,
            )

        self.predictor = SAM2ImagePredictor(self.model)

    def segment_image(
        self,
        image_rgb,
        boxes: List[List[float]]
    ) -> List[Tuple]:
        """
        Segment objects in an image given bounding boxes.

        Args:
            image_rgb: np.ndarray (H, W, 3), RGB image
            boxes: List of bounding boxes in pixel coordinates
                   format: [x1, y1, x2, y2]

        Returns:
            List of tuples:
                (mask, score, box)

            where:
                mask: np.ndarray (H, W), binary mask
                score: float, confidence score
                box: np.ndarray (4,), bounding box used
        """
        results = []
        
        # Set image once for predictor
        self.predictor.set_image(image_rgb)

        for box in boxes:
            box_tensor  = torch.tensor([box], device=self.device)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_tensor,
                multimask_output=False,
            )
            results.append((masks[0], scores[0], box_tensor[0].cpu().numpy()))

        return results