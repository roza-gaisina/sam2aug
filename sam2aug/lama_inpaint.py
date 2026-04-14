"""
LaMa inpainting implementation.

This module integrates the LaMa (Large Mask Inpainting) model into the pipeline
and provides a callable interface for performing inpainting.

Source:
Adapted from:
https://github.com/geekyutao/Inpaint-Anything/blob/main/lama_inpaint.py

Modifications:
- Integrated into a reusable pipeline component
- Simplified for direct Python usage (no subprocess calls)

This module is implementation-specific and should not be modified unless
the inpainting backend changes.
"""

import os
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo

# Limit CPU threading for stability
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

@torch.no_grad()
def inpaint_img_with_lama(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    model,
    device="cuda",
    mod=8
) -> np.ndarray:
    """
    Inpaint an image using LaMa.

    Args:
        image_bgr: np.ndarray (H, W, 3), BGR image
        mask: np.ndarray (H, W), binary mask (0/1 or 0/255)
        model: loaded LaMa model
        device: torch device
        mod: padding modulo

    Returns:
        np.ndarray (H, W, 3), inpainted BGR image
    """
    assert len(mask.shape) == 2

    if np.max(mask) == 1:
        mask = mask * 255

    image = torch.from_numpy(image_bgr).float().div(255.)
    mask = torch.from_numpy(mask).float()

    batch = {
        'image': image.permute(2, 0, 1).unsqueeze(0),
        'mask': mask[None, None]
    }

    original_size = batch['image'].shape[2:]

    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)

    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    result = batch["inpainted"][0].permute(1, 2, 0).cpu().numpy()

    result = result[:original_size[0], :original_size[1]]

    return np.clip(result * 255, 0, 255).astype(np.uint8)


def build_lama_model(        
        config_p: str,
        ckpt_p: str,
        device="cuda"
):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(train_config, checkpoint_path, strict=False)
    model.to(device)
    model.freeze()
    return model