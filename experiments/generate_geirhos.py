#!/usr/bin/env python3
"""
Geirhos-style dataset generation for your thesis.

Single entrypoint with methods:
  - texture_only
  - texture_plus_edges  (Sobel edges from original object added into donor fill)

Config-driven with optional YAML inheritance.

Outputs go into:
  <paths.outputs.root>/<experiment.name>/
    images/...
    meta/geirhos_gen_manifest.json

Run:
  python generate_geirhos.py --config configs/geirhos_texture_only.yaml --method texture_only
  python generate_geirhos.py --config configs/geirhos_texture_plus_edges.yaml --method texture_plus_edges
  python generate_geirhos.py --config configs/geirhos_texture_nst.yaml --method style_transfer --overwrite
  python generate_geirhos.py --config configs/geirhos_texture_adain.yaml --method texture_adain --overwrite

Expected config keys (typical):
  experiment:
    name: geirhos_texture_only

  paths:
    imagenet:
      val_root: /data/local/.../imagenet/val_classed
    outputs:
      root: /data/local/.../experiments

  content:
    manifest: /data/local/.../meta/val_selected_manifest.json

  textures:
    donor_dir: /data/local/.../donor_images_geirhos_square
    texture_wnids:
      - n02128385
      - n01806143
      ...

  generation:
    resize_mode: cover        # cover|stretch
    placement: full_image     # full_image|object_bbox_tight
    feather_px: 1
    dilate_px: 2
    jpeg_quality: 95

  edges:                      # only used for texture_plus_edges
    enabled: true             # optional
    strength: 2.0
    sobel_ksize: 3
    blur_ksize: 5
    blur_sigma: 1.0
    edge_band_px: 0
    edge_gamma: 1.0

Notes:
- Donor file names are expected as <wnid>.{jpg|jpeg|png|webp} inside donor_dir.
- Generated manifest entries include both shape_wnid and texture_wnid, plus output_rel_path.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import torchvision.models as models

import numpy as np
from PIL import Image, ImageFilter
import torch.nn as nn

# AdaIN repo (for texture_adain method)
ADAIN_REPO = "/home/stud/rgaisina/rg-master-thesis/pytorch-AdaIN"
VGG_PATH = "/home/stud/rgaisina/rg-master-thesis/models/adain/vgg_normalised.pth"
DEC_PATH = "/home/stud/rgaisina/rg-master-thesis/models/adain/decoder.pth"

# Sobel edges
import cv2

# YAML
try:
    import yaml
except ImportError as e:
    raise ImportError("Missing dependency 'pyyaml'. Install with: pip install pyyaml") from e

# Your pipeline
from sam2aug.segmenter import Segmenter
import sam2aug.config as sam2cfg

import sys
if ADAIN_REPO not in sys.path:
    sys.path.append(ADAIN_REPO)
from net import vgg, decoder
from function import adaptive_instance_normalization


# ----------------------------
# Config helpers
# ----------------------------
def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping/dict at top level.")
    return data


def load_config_with_inherits(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    parent = cfg.get("inherits")
    if not parent:
        return cfg

    if not isinstance(parent, str):
        raise ValueError("'inherits' must be a string path to a YAML file.")

    parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(config_path), parent)
    base_cfg = load_yaml(parent_path)
    child_cfg = dict(cfg)
    child_cfg.pop("inherits", None)
    return deep_update(base_cfg, child_cfg)


# ----------------------------
# Image / mask utilities
# ----------------------------
def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_rgb(path: str, rgb: np.ndarray, quality: int = 95) -> None:
    Image.fromarray(rgb).save(path, quality=quality)


def get_mask_from_segmenter(segmenter: Segmenter, image_rgb: np.ndarray, bbox_xyxy: List[int]) -> np.ndarray:
    """
    Segmenter.segment_image(image_rgb, boxes) returns list of (mask, score, box_np).
    We take the first mask.
    """
    segs = segmenter.segment_image(image_rgb, [bbox_xyxy])
    if not segs:
        return np.zeros(image_rgb.shape[:2], dtype=bool)

    mask, score, _ = segs[0]

    if hasattr(mask, "detach"):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.array(mask)

    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
        mask_np = mask_np[0]

    if mask_np.dtype == np.bool_:
        return mask_np

    return mask_np > 0.5


def dilate_mask_pil(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return mask
    k = max(3, 2 * dilate_px + 1)
    m = (mask.astype(np.uint8) * 255)
    m_img = Image.fromarray(m).filter(ImageFilter.MaxFilter(size=k))
    return (np.array(m_img) > 127)


def soft_alpha_from_mask(mask: np.ndarray, feather_px: int) -> np.ndarray:
    m = (mask.astype(np.uint8) * 255)
    m_img = Image.fromarray(m)
    if feather_px > 0:
        m_img = m_img.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return (np.array(m_img).astype(np.float32) / 255.0)


def compute_tight_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2, y2


def resize_donor_to_target(donor_rgb: np.ndarray, target_hw: Tuple[int, int], resize_mode: str) -> np.ndarray:
    """
    resize_mode:
      - stretch: resize donor exactly to target (may distort aspect)
      - cover: preserve aspect; scale to cover target then center-crop (no borders)
    """
    th, tw = target_hw
    donor_img = Image.fromarray(donor_rgb)

    if resize_mode == "stretch":
        return np.array(donor_img.resize((tw, th), resample=Image.BICUBIC))

    if resize_mode == "cover":
        dh, dw = donor_rgb.shape[:2]
        scale = max(tw / dw, th / dh)
        nw, nh = int(round(dw * scale)), int(round(dh * scale))
        resized = donor_img.resize((nw, nh), resample=Image.BICUBIC)
        left = max(0, (nw - tw) // 2)
        top = max(0, (nh - th) // 2)
        cropped = resized.crop((left, top, left + tw, top + th))
        return np.array(cropped)

    raise ValueError(f"Unknown resize_mode: {resize_mode}")


def blend_inside_mask(content_rgb: np.ndarray, donor_rgb_matched: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float32)
    out = content_rgb.astype(np.float32) * (1.0 - a) + donor_rgb_matched.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def resolve_donor_path(donor_dir: str, wnid: str) -> str:
    exts = [".jpg", ".jpeg", ".png", ".webp"]
    for ext in exts:
        p = os.path.join(donor_dir, wnid + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Donor not found for '{wnid}' in {donor_dir}. Expected: {', '.join(wnid + e for e in exts)}"
    )


# ----------------------------
# Sobel edges utilities
# ----------------------------
def make_edge_band(mask_bool: np.ndarray, band_px: int) -> np.ndarray:
    """
    band_px=0 -> full object mask
    band_px>0 -> boundary band (dilate XOR erode)
    """
    if band_px <= 0:
        return mask_bool

    m = mask_bool.astype(np.uint8)
    k = max(3, 2 * band_px + 1)
    kernel = np.ones((k, k), np.uint8)
    dil = cv2.dilate(m, kernel, iterations=1)
    ero = cv2.erode(m, kernel, iterations=1)
    band = (dil > 0) & (ero == 0)
    return band


def sobel_edges_rgb01(
    content_rgb: np.ndarray,
    mask_bool: np.ndarray,
    strength: float,
    sobel_ksize: int,
    blur_ksize: int,
    blur_sigma: float,
    edge_band_px: int,
    edge_gamma: float,
) -> np.ndarray:
    """
    Returns RGB edges in float32 [0,1], already masked to object/band and blurred.
    """
    gray = cv2.cvtColor(content_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    ksize = int(sobel_ksize)
    if ksize not in [1, 3, 5, 7]:
        raise ValueError("sobel_ksize should be one of {1,3,5,7}")

    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(sx * sx + sy * sy)

    mmax = float(mag.max())
    mag = mag / (mmax + 1e-8) if mmax > 1e-8 else np.zeros_like(mag)

    band = make_edge_band(mask_bool, int(edge_band_px))
    mag = mag * band.astype(np.float32)

    if edge_gamma != 1.0:
        mag = np.power(np.clip(mag, 0.0, 1.0), float(edge_gamma))

    bk = int(blur_ksize)
    if bk % 2 == 0:
        bk += 1
    if bk < 3:
        bk = 3

    mag = cv2.GaussianBlur(mag, (bk, bk), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))
    #mag = np.clip(mag * float(strength), 0.0, 1.0)
    mag = mag * float(strength)  # no clip here (match notebook behavior)

    return np.repeat(mag[:, :, None], 3, axis=2).astype(np.float32)


def add_edges_to_donor(donor_full_rgb: np.ndarray, edges_rgb01: np.ndarray) -> np.ndarray:
    donor_f = donor_full_rgb.astype(np.float32) / 255.0
    out = np.clip(donor_f + edges_rgb01, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)

# ----------------------------
# Style Transfer helpers
# ----------------------------
_VGG_CACHE = {"model": None, "device": None}
_ADAIN_CACHE = {"vgg_trunc": None, "decoder": None, "device": None}

def _get_adain_models(device: torch.device):
    if _ADAIN_CACHE["vgg_trunc"] is not None and _ADAIN_CACHE["device"] == str(device):
        return _ADAIN_CACHE["vgg_trunc"], _ADAIN_CACHE["decoder"]

    vgg.load_state_dict(torch.load(VGG_PATH, map_location=device))
    decoder.load_state_dict(torch.load(DEC_PATH, map_location=device))

    vgg_trunc = nn.Sequential(*list(vgg.children())[:31]).to(device).eval()
    dec_net = decoder.to(device).eval()

    for p in vgg_trunc.parameters():
        p.requires_grad_(False)
    for p in dec_net.parameters():
        p.requires_grad_(False)

    _ADAIN_CACHE["vgg_trunc"] = vgg_trunc
    _ADAIN_CACHE["decoder"] = dec_net
    _ADAIN_CACHE["device"] = str(device)
    return vgg_trunc, dec_net

def _pil_to_tensor_square(pil: Image.Image, device: torch.device, size: int = 384) -> torch.Tensor:
    pil = pil.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(pil)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)

@torch.no_grad()
def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((t * 255).astype(np.uint8))

@torch.no_grad()
def adain_transfer_crop(
    content_crop_pil: Image.Image,
    style_pil: Image.Image,
    device: torch.device,
    alpha: float = 1.0,
    size: int = 256,
) -> Image.Image:
    vgg_trunc, dec_net = _get_adain_models(device)
    orig_w, orig_h = content_crop_pil.size

    content = _pil_to_tensor_square(content_crop_pil, device=device, size=size)
    style = _pil_to_tensor_square(style_pil, device=device, size=size)

    c_feat = vgg_trunc(content)
    s_feat = vgg_trunc(style)

    t = adaptive_instance_normalization(c_feat, s_feat)
    t = alpha * t + (1 - alpha) * c_feat

    out = dec_net(t)
    out_pil = _tensor_to_pil(out)
    return out_pil.resize((orig_w, orig_h), Image.BILINEAR)

@torch.no_grad()
def adain_transfer_on_segmented_object(
    content_rgb: np.ndarray,
    style_rgb: np.ndarray,
    segmenter: Segmenter,
    bbox_xyxy: List[int],
    device: torch.device,
    adain_alpha: float = 1.0,
    size: int = 256,
    dilate_px: int = 2,
    feather_px: int = 2,
) -> Dict[str, Any]:
    mask_bool = get_mask_from_segmenter(segmenter, content_rgb, bbox_xyxy)

    if dilate_px > 0:
        mask_bool = dilate_mask_pil(mask_bool, dilate_px)

    alpha_mask = soft_alpha_from_mask(mask_bool, feather_px)

    x1, y1, x2, y2 = compute_tight_bbox_from_mask(mask_bool)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Segmentation mask is empty. Check bbox / XML annotation.")

    content_crop_pil = Image.fromarray(content_rgb[y1:y2, x1:x2])
    style_pil = Image.fromarray(style_rgb)
    stylized_crop_pil = adain_transfer_crop(
        content_crop_pil=content_crop_pil,
        style_pil=style_pil,
        device=device,
        alpha=adain_alpha,
        size=size,
    )

    stylized_crop_rgb = np.array(stylized_crop_pil)
    stylized_full_rgb = content_rgb.copy()
    stylized_full_rgb[y1:y2, x1:x2] = stylized_crop_rgb
    final_rgb = blend_inside_mask(content_rgb, stylized_full_rgb, alpha_mask)

    return {
        "output_rgb": final_rgb,
        "mask_bool": mask_bool,
        "alpha_mask": alpha_mask,
        "crop_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "stylized_crop_rgb": stylized_crop_rgb,
        "input_bbox_xyxy": [int(v) for v in bbox_xyxy],
    }
    
def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return (x - mean) / std

def _to_tensor01_rgb(img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img_rgb).float() / 255.0  # HWC
    x = x.permute(2,0,1).unsqueeze(0).to(device)   # 1CHW
    return x

def _to_uint8_rgb(x01: torch.Tensor) -> np.ndarray:
    x = x01.detach().clamp(0,1)[0].permute(1,2,0).cpu().numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)

def _gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    # feat: 1xCxHxW
    b, c, h, w = feat.shape
    f = feat.view(c, h*w)
    g = f @ f.t()
    return g / (c * h * w + 1e-8)

def _get_vgg19(device: torch.device):
    if _VGG_CACHE["model"] is not None and _VGG_CACHE["device"] == str(device):
        return _VGG_CACHE["model"]

    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    _VGG_CACHE["model"] = vgg
    _VGG_CACHE["device"] = str(device)
    return vgg

_LAYER_MAP = {
    # VGG19 features indices
    "relu1_1": 1,
    "relu2_1": 6,
    "relu3_1": 11,
    "relu4_1": 20,
    "relu4_2": 22,
    "relu5_1": 29,
}

def _vgg_extract(vgg, x_norm: torch.Tensor, layer_names: List[str]) -> Dict[str, torch.Tensor]:
    targets = {name: _LAYER_MAP[name] for name in layer_names}
    max_i = max(targets.values())
    out: Dict[str, torch.Tensor] = {}
    h = x_norm
    for i, layer in enumerate(vgg):
        h = layer(h)
        for name, idx in targets.items():
            if i == idx:
                out[name] = h
        if i >= max_i:
            break
    return out

def neural_style_transfer_crop(
    content_crop_rgb: np.ndarray,
    style_rgb: np.ndarray,
    device: torch.device,
    out_size: int = 384,
    iters: int = 300,
    style_weight: float = 5e4,
    content_weight: float = 0.2,
    tv_weight: float = 1e-4,
    init: str = "content",  # "content" | "noise"
    style_layers: List[str] = None,
    content_layer: str = "relu4_1",
    style_layer_weights: Dict[str, float] = None,
    lbfgs_inner_iters: int = 20,
    clamp_each_step: bool = True,
) -> np.ndarray:
    """
    Optimization-based NST (Gatys) on a crop with *square resize always* (out_size x out_size),
    using VGG19 features, per-layer style weights, and LBFGS.

    Returns: stylized crop as uint8 RGB with the SAME spatial size as the original crop.
    """
    if style_layers is None:
        style_layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]

    if style_layer_weights is None:
        # Default: emphasize shallow layers (texture-heavy), similar to your pilot script
        style_layer_weights = {
            "relu1_1": 1.0,
            "relu2_1": 0.8,
            "relu3_1": 0.3,
            "relu4_1": 0.1,
            "relu5_1": 0.05,
        }

    # ---- square resize always (match pilot) ----
    ch, cw = content_crop_rgb.shape[:2]
    content_sq = np.array(
        Image.fromarray(content_crop_rgb).resize((out_size, out_size), resample=Image.BILINEAR)
    )
    style_sq = np.array(
        Image.fromarray(style_rgb).resize((out_size, out_size), resample=Image.BILINEAR)
    )

    vgg = _get_vgg19(device)

    # Prepare tensors
    content = _to_tensor01_rgb(content_sq, device)  # 1x3xSxS
    style = _to_tensor01_rgb(style_sq, device)

    if init == "noise":
        target = torch.rand_like(content, requires_grad=True)
    else:
        target = content.clone().detach().requires_grad_(True)

    # Precompute targets
    needed_layers = list(set(style_layers + [content_layer]))
    with torch.no_grad():
        c_feats = _vgg_extract(vgg, _imagenet_normalize(content), [content_layer])[content_layer]
        s_feats = _vgg_extract(vgg, _imagenet_normalize(style), style_layers)
        s_grams = {k: _gram_matrix(v) for k, v in s_feats.items()}

    # TV loss (same as your pilot)
    def _tv_loss(x: torch.Tensor) -> torch.Tensor:
        return (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() + (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()

    # LBFGS (match pilot: max_iter=20; run rounds)
    inner = max(1, int(lbfgs_inner_iters))
    rounds = max(1, int(iters) // inner)

    opt = torch.optim.LBFGS([target], max_iter=inner, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)

        t_feats = _vgg_extract(vgg, _imagenet_normalize(target), needed_layers)

        lc = F.mse_loss(t_feats[content_layer], c_feats)

        ls = 0.0
        for k, g in s_grams.items():
            tg = _gram_matrix(t_feats[k])
            w = float(style_layer_weights.get(k, 1.0))
            ls = ls + w * F.mse_loss(tg, g)

        tv = _tv_loss(target)

        loss = content_weight * lc + style_weight * ls + tv_weight * tv
        loss.backward()

        # ---- FIX for LBFGS: make grad contiguous ----
        if target.grad is not None and (not target.grad.is_contiguous()):
            target.grad = target.grad.contiguous()

        if clamp_each_step:
            with torch.no_grad():
                target.clamp_(0.0, 1.0)

        return loss


    for _ in range(rounds):
        opt.step(closure)

    # Convert stylized square back to original crop size (cw x ch)
    out_sq = _to_uint8_rgb(target)
    out_full = np.array(Image.fromarray(out_sq).resize((cw, ch), resample=Image.BILINEAR))
    return out_full

# ----------------------------
# Core generation routines
# ----------------------------
def _common_init(cfg: Dict[str, Any]):
    # required
    val_root = cfg["paths"]["imagenet"]["val_root"]
    out_root = os.path.join(cfg["paths"]["outputs"]["root"], cfg["experiment"]["name"])
    content_manifest = cfg["content"]["manifest"]

    textures = cfg["textures"]
    donor_dir = textures["donor_dir"]
    texture_wnids = textures["texture_wnids"]

    gen = cfg.get("generation", {})
    resize_mode = gen.get("resize_mode", "cover")
    placement = gen.get("placement", "full_image")
    feather_px = int(gen.get("feather_px", 1))
    dilate_px = int(gen.get("dilate_px", 0))
    jpeg_quality = int(gen.get("jpeg_quality", 95))

    # init segmenter
    segmenter = Segmenter(
        model_config=sam2cfg.SAM2_CONFIG,
        checkpoint_path=sam2cfg.SAM2_CHECKPOINT,
        device=sam2cfg.DEVICE,
    )

    # load content manifest
    with open(content_manifest, "r") as f:
        base = json.load(f)

    # preload donors
    donor_paths: Dict[str, str] = {}
    donor_cache: Dict[str, np.ndarray] = {}
    for twnid in texture_wnids:
        p = resolve_donor_path(donor_dir, twnid)
        donor_paths[twnid] = p
        donor_cache[twnid] = load_rgb(p)

    return {
        "val_root": val_root,
        "out_root": out_root,
        "content_manifest": content_manifest,
        "base": base,
        "segmenter": segmenter,
        "donor_paths": donor_paths,
        "donor_cache": donor_cache,
        "texture_wnids": texture_wnids,
        "resize_mode": resize_mode,
        "placement": placement,
        "feather_px": feather_px,
        "dilate_px": dilate_px,
        "jpeg_quality": jpeg_quality,
    }


def generate_texture_only(cfg: Dict[str, Any], overwrite: bool = False, save_debug: bool = False):
    ctx = _common_init(cfg)
    texture_names = cfg["textures"].get("texture_names", {})

    out_root = ctx["out_root"]
    out_images_dir = os.path.join(out_root, "images")
    out_meta_dir = os.path.join(out_root, "meta")
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)
    Path(out_meta_dir).mkdir(parents=True, exist_ok=True)

    base = ctx["base"]
    segmenter = ctx["segmenter"]

    gen_manifest = []
    total = 0

    print("[INFO] Generating TEXTURE-ONLY dataset")
    print("[INFO] out_root:", out_root)
    print("[INFO] N content:", len(base), "| N textures:", len(ctx["texture_wnids"]))
    print("[INFO] resize_mode:", ctx["resize_mode"], "placement:", ctx["placement"],
          "feather_px:", ctx["feather_px"], "dilate_px:", ctx["dilate_px"])

    for item in base:
        shape_wnid = item["wnid"]
        shape_name = item["class_name"]
        

        content_rel = item["original_rel_path"]
        content_path = os.path.join(ctx["val_root"], content_rel)
        content = load_rgb(content_path)
        H, W = content.shape[:2]

        bbox = item["primary_bbox_xyxy"]
        mask_bool = get_mask_from_segmenter(segmenter, content, bbox)
        if ctx["dilate_px"] > 0:
            mask_bool = dilate_mask_pil(mask_bool, ctx["dilate_px"])
        alpha = soft_alpha_from_mask(mask_bool, ctx["feather_px"])

        if ctx["placement"] == "full_image":
            target_hw = (H, W)
            donor_target_bbox = None
        else:
            x1, y1, x2, y2 = compute_tight_bbox_from_mask(mask_bool)
            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = 0, 0, W, H
            target_hw = (y2 - y1, x2 - x1)
            donor_target_bbox = (x1, y1, x2, y2)

        for tex_wnid in ctx["texture_wnids"]:
            if tex_wnid == shape_wnid:
                continue

            donor = ctx["donor_cache"][tex_wnid]
            donor_matched = resize_donor_to_target(donor, target_hw, ctx["resize_mode"])

            if ctx["placement"] == "full_image":
                donor_full = donor_matched
            else:
                donor_full = content.copy()
                x1, y1, x2, y2 = donor_target_bbox
                donor_full[y1:y2, x1:x2] = donor_matched

            out_rel = os.path.join(
                "images",
                shape_wnid,
                f"{item['image_id']}__shape-{shape_wnid}__tex-{tex_wnid}.jpg",
            )
            out_path = os.path.join(out_root, out_rel)
            Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

            if overwrite or (not os.path.exists(out_path)):
                out = blend_inside_mask(content, donor_full, alpha)
                save_rgb(out_path, out, quality=ctx["jpeg_quality"])
                total += 1

            gen_manifest.append({
                "image_id": item["image_id"],
                "shape_wnid": shape_wnid,
                "shape_name": shape_name,
                "texture_wnid": tex_wnid,
                "texture_name": texture_names.get(tex_wnid, tex_wnid),
                "content_rel_path": content_rel,
                "primary_bbox_xyxy": bbox,
                "image_size_hw": item.get("image_size_hw", [H, W]),
                "donor_path": ctx["donor_paths"][tex_wnid],
                "resize_mode": ctx["resize_mode"],
                "placement": ctx["placement"],
                "feather_px": ctx["feather_px"],
                "dilate_px": ctx["dilate_px"],
                "output_rel_path": out_rel,
                "method": "texture_only",
            })

        if save_debug:
            dbg_dir = os.path.join(out_root, "debug", shape_wnid)
            Path(dbg_dir).mkdir(parents=True, exist_ok=True)
            Image.fromarray((alpha * 255).astype(np.uint8)).save(os.path.join(dbg_dir, f"{item['image_id']}__alpha.png"))
            Image.fromarray((mask_bool.astype(np.uint8) * 255).astype(np.uint8)).save(os.path.join(dbg_dir, f"{item['image_id']}__mask.png"))

    out_manifest_path = os.path.join(out_meta_dir, "geirhos_gen_manifest.json")
    with open(out_manifest_path, "w") as f:
        json.dump(gen_manifest, f, indent=2)

    print(f"[INFO] Done. Wrote {total} images.")
    print("[INFO] Manifest:", out_manifest_path)


def generate_texture_plus_edges(cfg: Dict[str, Any], overwrite: bool = False, save_debug: bool = False):
    ctx = _common_init(cfg)
    texture_names = cfg["textures"].get("texture_names", {})

    edges_cfg = cfg.get("edges", {}) or {}
    enabled = bool(edges_cfg.get("enabled", True))
    if not enabled:
        raise ValueError("edges.enabled is false, but you invoked method=texture_plus_edges")

    strength = float(edges_cfg.get("strength", 2.0))
    sobel_ksize = int(edges_cfg.get("sobel_ksize", 3))
    blur_ksize = int(edges_cfg.get("blur_ksize", 5))
    blur_sigma = float(edges_cfg.get("blur_sigma", 1.0))
    edge_band_px = int(edges_cfg.get("edge_band_px", 0))
    edge_gamma = float(edges_cfg.get("edge_gamma", 1.0))

    out_root = ctx["out_root"]
    out_images_dir = os.path.join(out_root, "images")
    out_meta_dir = os.path.join(out_root, "meta")
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)
    Path(out_meta_dir).mkdir(parents=True, exist_ok=True)

    base = ctx["base"]
    segmenter = ctx["segmenter"]

    gen_manifest = []
    total = 0

    print("[INFO] Generating TEXTURE+EDGES dataset")
    print("[INFO] out_root:", out_root)
    print("[INFO] N content:", len(base), "| N textures:", len(ctx["texture_wnids"]))
    print("[INFO] resize_mode:", ctx["resize_mode"], "placement:", ctx["placement"],
          "feather_px:", ctx["feather_px"], "dilate_px:", ctx["dilate_px"])
    print("[INFO] edges:", dict(
        strength=strength, sobel_ksize=sobel_ksize,
        blur_ksize=blur_ksize, blur_sigma=blur_sigma,
        edge_band_px=edge_band_px, edge_gamma=edge_gamma
    ))

    for item in base:
        shape_wnid = item["wnid"]
        shape_name = item["class_name"]

        content_rel = item["original_rel_path"]
        content_path = os.path.join(ctx["val_root"], content_rel)
        content = load_rgb(content_path)
        H, W = content.shape[:2]

        bbox = item["primary_bbox_xyxy"]
        mask_bool = get_mask_from_segmenter(segmenter, content, bbox)
        if ctx["dilate_px"] > 0:
            mask_bool = dilate_mask_pil(mask_bool, ctx["dilate_px"])
        alpha = soft_alpha_from_mask(mask_bool, ctx["feather_px"])

        # edges once per content image
        edges_rgb01 = sobel_edges_rgb01(
            content_rgb=content,
            mask_bool=mask_bool,
            strength=strength,
            sobel_ksize=sobel_ksize,
            blur_ksize=blur_ksize,
            blur_sigma=blur_sigma,
            edge_band_px=edge_band_px,
            edge_gamma=edge_gamma,
        )

        if ctx["placement"] == "full_image":
            target_hw = (H, W)
            donor_target_bbox = None
        else:
            x1, y1, x2, y2 = compute_tight_bbox_from_mask(mask_bool)
            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = 0, 0, W, H
            target_hw = (y2 - y1, x2 - x1)
            donor_target_bbox = (x1, y1, x2, y2)

        for tex_wnid in ctx["texture_wnids"]:
            if tex_wnid == shape_wnid:
                continue

            donor = ctx["donor_cache"][tex_wnid]
            donor_matched = resize_donor_to_target(donor, target_hw, ctx["resize_mode"])

            if ctx["placement"] == "full_image":
                donor_full = donor_matched
            else:
                donor_full = content.copy()
                x1, y1, x2, y2 = donor_target_bbox
                donor_full[y1:y2, x1:x2] = donor_matched

            donor_with_edges = add_edges_to_donor(donor_full, edges_rgb01)

            out_rel = os.path.join(
                "images",
                shape_wnid,
                f"{item['image_id']}__shape-{shape_wnid}__tex-{tex_wnid}__edges.jpg",
            )
            out_path = os.path.join(out_root, out_rel)
            Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

            if overwrite or (not os.path.exists(out_path)):
                out = blend_inside_mask(content, donor_with_edges, alpha)
                save_rgb(out_path, out, quality=ctx["jpeg_quality"])
                total += 1

            gen_manifest.append({
                "image_id": item["image_id"],
                "shape_wnid": shape_wnid,
                "shape_name": shape_name,
                "texture_wnid": tex_wnid,
                "texture_name": texture_names.get(tex_wnid, tex_wnid),
                "content_rel_path": content_rel,
                "primary_bbox_xyxy": bbox,
                "image_size_hw": item.get("image_size_hw", [H, W]),
                "donor_path": ctx["donor_paths"][tex_wnid],
                "resize_mode": ctx["resize_mode"],
                "placement": ctx["placement"],
                "feather_px": ctx["feather_px"],
                "dilate_px": ctx["dilate_px"],
                "edges": {
                    "strength": strength,
                    "sobel_ksize": sobel_ksize,
                    "blur_ksize": blur_ksize,
                    "blur_sigma": blur_sigma,
                    "edge_band_px": edge_band_px,
                    "edge_gamma": edge_gamma,
                },
                "output_rel_path": out_rel,
                "method": "texture_plus_edges",
            })

        if save_debug:
            dbg_dir = os.path.join(out_root, "debug", shape_wnid)
            Path(dbg_dir).mkdir(parents=True, exist_ok=True)
            Image.fromarray((alpha * 255).astype(np.uint8)).save(os.path.join(dbg_dir, f"{item['image_id']}__alpha.png"))
            Image.fromarray((mask_bool.astype(np.uint8) * 255).astype(np.uint8)).save(os.path.join(dbg_dir, f"{item['image_id']}__mask.png"))
            eprev = (np.clip(edges_rgb01, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(eprev).save(os.path.join(dbg_dir, f"{item['image_id']}__edges_preview.png"))

    out_manifest_path = os.path.join(out_meta_dir, "geirhos_gen_manifest.json")
    with open(out_manifest_path, "w") as f:
        json.dump(gen_manifest, f, indent=2)

    print(f"[INFO] Done. Wrote {total} images.")
    print("[INFO] Manifest:", out_manifest_path)

def generate_style_transfer(cfg: Dict[str, Any], overwrite: bool = False, save_debug: bool = False):
    """
    3rd method: optimization-based NST (Gatys) applied to the object crop (tight bbox),
    with square resize always (handled inside neural_style_transfer_crop).

    Expects config keys (recommended):
      nst:
        enabled: true
        out_size: 384
        iters: 300
        lbfgs_inner_iters: 20
        style_weight: 50000
        content_weight: 0.2
        tv_weight: 1e-4
        init: content
        style_layer_weights: {relu1_1: 1.0, ...}
    """
    ctx = _common_init(cfg)

    # ---- read NST config ----
    nst = (cfg.get("nst") or cfg.get("style_transfer") or {})  # allow legacy key if you used it
    if not bool(nst.get("enabled", True)):
        raise ValueError("nst.enabled is false, but you invoked method=style_transfer")

    out_size = int(nst.get("out_size", 384))
    iters = int(nst.get("iters", 300))
    lbfgs_inner_iters = int(nst.get("lbfgs_inner_iters", 20))
    style_weight = float(nst.get("style_weight", 5e4))
    content_weight = float(nst.get("content_weight", 0.2))
    tv_weight = float(nst.get("tv_weight", 1e-4))
    init = str(nst.get("init", "content"))

    # style layer weights: keys may be "relu1_1" etc. (preferred)
    style_layer_weights = nst.get("style_layer_weights", None)
    if isinstance(style_layer_weights, dict):
        # ensure float values
        style_layer_weights = {str(k): float(v) for k, v in style_layer_weights.items()}
    else:
        style_layer_weights = None

    # Use CUDA if available (NST is heavy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = ctx["out_root"]
    out_images_dir = os.path.join(out_root, "images")
    out_meta_dir = os.path.join(out_root, "meta")
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)
    Path(out_meta_dir).mkdir(parents=True, exist_ok=True)

    base = ctx["base"]
    segmenter = ctx["segmenter"]

    gen_manifest = []
    total_saved = 0

    print("[INFO] Generating STYLE-TRANSFER (NST) dataset")
    print("[INFO] out_root:", out_root)
    print("[INFO] NST device:", device)
    print("[INFO] NST params:", {
        "out_size": out_size,
        "iters": iters,
        "lbfgs_inner_iters": lbfgs_inner_iters,
        "style_weight": style_weight,
        "content_weight": content_weight,
        "tv_weight": tv_weight,
        "init": init,
        "style_layer_weights": style_layer_weights,
    })

    tex_names = (cfg.get("textures", {}) or {}).get("texture_names", {}) or {}

    for item in base:
        shape_wnid = item["wnid"]
        shape_name = item.get("class_name", shape_wnid)

        content_rel = item["original_rel_path"]
        content_path = os.path.join(ctx["val_root"], content_rel)
        content = load_rgb(content_path)
        H, W = content.shape[:2]

        bbox = item["primary_bbox_xyxy"]
        mask_bool = get_mask_from_segmenter(segmenter, content, bbox)
        if ctx["dilate_px"] > 0:
            mask_bool = dilate_mask_pil(mask_bool, ctx["dilate_px"])

        # soft alpha for blending into full image
        alpha = soft_alpha_from_mask(mask_bool, ctx["feather_px"])

        # tight bbox crop for NST
        x1, y1, x2, y2 = compute_tight_bbox_from_mask(mask_bool)
        if x2 <= x1 or y2 <= y1:
            # fallback: whole image if mask fails
            x1, y1, x2, y2 = 0, 0, W, H

        content_crop = content[y1:y2, x1:x2]
        mask_crop = mask_bool[y1:y2, x1:x2]
        alpha_crop = alpha[y1:y2, x1:x2]

        for tex_wnid in ctx["texture_wnids"]:
            if tex_wnid == shape_wnid:
                continue

            # donor image (full texture image)
            donor_full = ctx["donor_cache"][tex_wnid]

            # ---- NST on crop (square resize happens inside) ----
            styl_crop = neural_style_transfer_crop(
                content_crop_rgb=content_crop,
                style_rgb=donor_full,
                device=device,
                out_size=out_size,
                iters=iters,
                style_weight=style_weight,
                content_weight=content_weight,
                tv_weight=tv_weight,
                init=init,
                style_layers=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
                content_layer="relu4_1",
                style_layer_weights=style_layer_weights,
                lbfgs_inner_iters=lbfgs_inner_iters,
                clamp_each_step=True,
            )

            # paste stylized crop back into a full-sized canvas
            styl_full = content.copy()
            styl_full[y1:y2, x1:x2] = styl_crop

            # blend only inside object mask (soft alpha over full image)
            out = blend_inside_mask(content, styl_full, alpha)

            out_rel = os.path.join(
                "images",
                shape_wnid,
                f"{item['image_id']}__shape-{shape_wnid}__tex-{tex_wnid}__nst.jpg",
            )
            out_path = os.path.join(out_root, out_rel)
            Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

            if overwrite or (not os.path.exists(out_path)):
                save_rgb(out_path, out, quality=ctx["jpeg_quality"])
                total_saved += 1

            gen_manifest.append({
                "image_id": item["image_id"],
                "shape_wnid": shape_wnid,
                "shape_name": shape_name,
                "texture_wnid": tex_wnid,
                "texture_name": tex_names.get(tex_wnid, tex_wnid),
                "content_rel_path": content_rel,
                "primary_bbox_xyxy": bbox,
                "image_size_hw": item.get("image_size_hw", [H, W]),
                "donor_path": ctx["donor_paths"][tex_wnid],
                "resize_mode": ctx["resize_mode"],
                "placement": "object_bbox_tight",
                "feather_px": ctx["feather_px"],
                "dilate_px": ctx["dilate_px"],
                "crop_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "mask_ratio_in_crop": float(mask_crop.mean()) if mask_crop.size else 0.0,
                "nst": {
                    "out_size": out_size,
                    "iters": iters,
                    "lbfgs_inner_iters": lbfgs_inner_iters,
                    "style_weight": style_weight,
                    "content_weight": content_weight,
                    "tv_weight": tv_weight,
                    "init": init,
                    "style_layer_weights": style_layer_weights,
                },
                "output_rel_path": out_rel,
                "method": "style_transfer",
            })

            if save_debug:
                dbg_dir = os.path.join(out_root, "debug", shape_wnid)
                Path(dbg_dir).mkdir(parents=True, exist_ok=True)
                base_id = item["image_id"]
                Image.fromarray((mask_crop.astype(np.uint8) * 255)).save(
                    os.path.join(dbg_dir, f"{base_id}__mask_crop.png")
                )
                Image.fromarray(styl_crop).save(
                    os.path.join(dbg_dir, f"{base_id}__styl_crop__tex-{tex_wnid}.jpg")
                )

    out_manifest_path = os.path.join(out_meta_dir, "geirhos_gen_manifest.json")
    with open(out_manifest_path, "w") as f:
        json.dump(gen_manifest, f, indent=2)

    print(f"[INFO] Done. Saved {total_saved} images.")
    print("[INFO] Manifest:", out_manifest_path)

def generate_texture_adain(cfg: Dict[str, Any], overwrite: bool = False, save_debug: bool = False):
    """
    AdaIN-based texture transfer applied only to the SAM2-segmented object.
    Uses the Geirhos16 content manifest and donor texture list from config.
    """
    ctx = _common_init(cfg)
    tex_names = (cfg.get("textures", {}) or {}).get("texture_names", {}) or {}

    adain_cfg = (cfg.get("adain") or cfg.get("texture_adain") or {})
    if not bool(adain_cfg.get("enabled", True)):
        raise ValueError("adain.enabled is false, but you invoked method=texture_adain")

    adain_alpha = float(adain_cfg.get("alpha", 1.0))
    adain_size = int(adain_cfg.get("size", 256))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = ctx["out_root"]
    out_images_dir = os.path.join(out_root, "images")
    out_meta_dir = os.path.join(out_root, "meta")
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)
    Path(out_meta_dir).mkdir(parents=True, exist_ok=True)

    base = ctx["base"]
    segmenter = ctx["segmenter"]

    gen_manifest = []
    total_saved = 0

    print("[INFO] Generating TEXTURE-AdaIN dataset")
    print("[INFO] out_root:", out_root)
    print("[INFO] AdaIN device:", device)
    print("[INFO] AdaIN params:", {
        "alpha": adain_alpha,
        "size": adain_size,
        "feather_px": ctx["feather_px"],
        "dilate_px": ctx["dilate_px"],
    })

    for item in base:
        shape_wnid = item["wnid"]
        shape_name = item.get("class_name", shape_wnid)

        content_rel = item["original_rel_path"]
        content_path = os.path.join(ctx["val_root"], content_rel)
        content = load_rgb(content_path)
        H, W = content.shape[:2]

        bbox = item["primary_bbox_xyxy"]

        for tex_wnid in ctx["texture_wnids"]:
            if tex_wnid == shape_wnid:
                continue

            donor_rgb = ctx["donor_cache"][tex_wnid]

            result = adain_transfer_on_segmented_object(
                content_rgb=content,
                style_rgb=donor_rgb,
                segmenter=segmenter,
                bbox_xyxy=bbox,
                device=device,
                adain_alpha=adain_alpha,
                size=adain_size,
                dilate_px=ctx["dilate_px"],
                feather_px=ctx["feather_px"],
            )
            out = result["output_rgb"]

            out_rel = os.path.join(
                "images",
                shape_wnid,
                f"{item['image_id']}__shape-{shape_wnid}__tex-{tex_wnid}__adain.jpg",
            )
            out_path = os.path.join(out_root, out_rel)
            Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

            if overwrite or (not os.path.exists(out_path)):
                save_rgb(out_path, out, quality=ctx["jpeg_quality"])
                total_saved += 1

            gen_manifest.append({
                "image_id": item["image_id"],
                "shape_wnid": shape_wnid,
                "shape_name": shape_name,
                "texture_wnid": tex_wnid,
                "texture_name": tex_names.get(tex_wnid, tex_wnid),
                "content_rel_path": content_rel,
                "primary_bbox_xyxy": bbox,
                "image_size_hw": item.get("image_size_hw", [H, W]),
                "donor_path": ctx["donor_paths"][tex_wnid],
                "placement": "object_bbox_tight",
                "feather_px": ctx["feather_px"],
                "dilate_px": ctx["dilate_px"],
                "crop_xyxy": result["crop_xyxy"],
                "adain": {
                    "alpha": adain_alpha,
                    "size": adain_size,
                },
                "output_rel_path": out_rel,
                "method": "texture_adain",
            })

            if save_debug:
                dbg_dir = os.path.join(out_root, "debug", shape_wnid)
                Path(dbg_dir).mkdir(parents=True, exist_ok=True)
                base_id = item["image_id"]
                Image.fromarray((result["mask_bool"].astype(np.uint8) * 255)).save(
                    os.path.join(dbg_dir, f"{base_id}__mask__tex-{tex_wnid}.png")
                )
                Image.fromarray((result["alpha_mask"] * 255).astype(np.uint8)).save(
                    os.path.join(dbg_dir, f"{base_id}__alpha__tex-{tex_wnid}.png")
                )
                Image.fromarray(result["stylized_crop_rgb"]).save(
                    os.path.join(dbg_dir, f"{base_id}__styl_crop__tex-{tex_wnid}.jpg")
                )

    out_manifest_path = os.path.join(out_meta_dir, "geirhos_gen_manifest.json")
    with open(out_manifest_path, "w") as f:
        json.dump(gen_manifest, f, indent=2)

    print(f"[INFO] Done. Saved {total_saved} images.")
    print("[INFO] Manifest:", out_manifest_path)

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=["texture_only", "texture_plus_edges", "style_transfer", "texture_adain"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save-debug", action="store_true")
    args = ap.parse_args()

    cfg = load_config_with_inherits(args.config)

    if args.method == "texture_only":
        generate_texture_only(cfg, overwrite=args.overwrite, save_debug=args.save_debug)
    elif args.method == "texture_plus_edges":
        generate_texture_plus_edges(cfg, overwrite=args.overwrite, save_debug=args.save_debug)
    elif args.method == "style_transfer":
        generate_style_transfer(cfg, overwrite=args.overwrite, save_debug=args.save_debug)
    elif args.method == "texture_adain":
        generate_texture_adain(cfg, overwrite=args.overwrite, save_debug=args.save_debug)
    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":
    main()