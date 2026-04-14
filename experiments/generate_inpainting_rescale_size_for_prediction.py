#!/usr/bin/env python3
"""
Generate same-background relocation images with controlled object scaling.

Experiment design
-----------------
- content images come from the curated Geirhos16 ImageNet subset manifest
- object is segmented from the original image
- original background is inpainted
- object is pasted back onto the SAME inpainted background
- placement is fixed to the center
- object scale is controlled relative to the extracted tight object crop

Recommended scale factors include:
  [1.0, 0.75, 0.50, 0.25]

Outputs go to:
  <paths.outputs.root>/<experiment.name>/
    images/<scale_tag>/<wnid>/<image_id>__scale-XXX.jpg
    meta/rescale_size_for_prediction_gen_manifest.json

Run:
  python generate_inpainting_rescale_size_for_prediction.py --config configs/inpainting_rescale_size_for_prediction.yaml --overwrite
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import yaml
except ImportError as e:
    raise ImportError("Missing dependency 'pyyaml'. Install with: pip install pyyaml") from e

from sam2aug.segmenter import Segmenter
import sam2aug.config as sam2cfg
from sam2aug.pipeline import AugmentationPipeline
from sam2aug.relocator import (
    Relocator,
    RelocatorConfig,
    PlacementConfig,
    ScaleConfig,
    BlendConfig,
)
from sam2aug.inpainter import LamaInpainter
from sam2aug.config import LAMA_CONFIG_PATH, LAMA_CKPT_PATH, DEVICE


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
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
    parent = cfg.get("inherits", None)
    if not parent:
        return cfg

    if not isinstance(parent, str):
        raise ValueError("'inherits' must be a string path to a YAML file.")

    parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(config_path), parent)
    base_cfg = load_yaml(parent_path)

    child_cfg = dict(cfg)
    child_cfg.pop("inherits", None)

    merged = deep_update(base_cfg, child_cfg)
    merged["_meta"] = {
        "config_path": config_path,
        "base_path": parent_path,
    }
    return merged

# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_rgb(path: str, rgb: np.ndarray, quality: int = 95) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=quality)


def to_u8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() == 1:
        mask = mask * 255
    return (mask > 0).astype(np.uint8) * 255


def tight_bbox_from_mask(mask_u8: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask_u8.shape[:2]
        return 0, 0, w, h
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def crop_object_tight(obj_rgb_full: np.ndarray, mask_u8_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = tight_bbox_from_mask(mask_u8_full)
    obj_crop = obj_rgb_full[y1:y2, x1:x2]
    mask_crop = mask_u8_full[y1:y2, x1:x2]
    return obj_crop, mask_crop, (x1, y1, x2, y2)


def resize_object_crop(obj_rgb: np.ndarray, mask_u8: np.ndarray, scale_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be > 0, got {scale_factor}")
    if abs(scale_factor - 1.0) < 1e-8:
        return obj_rgb, mask_u8

    h, w = obj_rgb.shape[:2]
    new_w = max(1, int(round(w * scale_factor)))
    new_h = max(1, int(round(h * scale_factor)))

    obj_r = np.array(Image.fromarray(obj_rgb).resize((new_w, new_h), resample=Image.BILINEAR))
    mask_r = np.array(Image.fromarray(mask_u8).resize((new_w, new_h), resample=Image.NEAREST))
    mask_r = to_u8_mask(mask_r)
    return obj_r, mask_r


def scale_tag(scale_factor: float) -> str:
    pct = int(round(scale_factor * 100.0))
    return f"scale{pct:03d}"


# -----------------------------------------------------------------------------
# Component builders
# -----------------------------------------------------------------------------
def build_segmenter() -> Segmenter:
    return Segmenter(
        model_config=sam2cfg.SAM2_CONFIG,
        checkpoint_path=sam2cfg.SAM2_CHECKPOINT,
        device=sam2cfg.DEVICE,
    )

def build_inpainter() -> LamaInpainter:
    return LamaInpainter(
        config_path=LAMA_CONFIG_PATH,
        ckpt_path=LAMA_CKPT_PATH,
        device=DEVICE,
    )

def _try_build_class(cls, kwargs_variants: Sequence[Dict[str, Any]]):
    errors = []
    for kwargs in kwargs_variants:
        try:
            return cls(**kwargs)
        except TypeError as e:
            errors.append(f"kwargs={kwargs}: {e}")
        except Exception as e:
            errors.append(f"kwargs={kwargs}: {type(e).__name__}: {e}")
    raise RuntimeError(" | ".join(errors))


def build_center_relocator(cfg: Dict[str, Any]) -> Relocator:
    rcfg = cfg.get("relocation", {}) or {}
    pcfg = rcfg.get("placement", {}) or {}
    bcfg = rcfg.get("blend", {}) or {}

    reloc_cfg = RelocatorConfig(
        placement=PlacementConfig(
            mode=str(pcfg.get("mode", "center")),
            anchor=str(pcfg.get("anchor", "center")),
            require_full_visibility=bool(pcfg.get("require_full_visibility", True)),
            margin_px=int(pcfg.get("margin_px", 0)),
            max_attempts=int(pcfg.get("max_attempts", 1)),
            jitter_px=int(pcfg.get("jitter_px", 0)),
        ),
        scale=ScaleConfig(
            mode="none",
            min_scale=1.0,
            max_scale=1.0,
            shrink_to_fit=True,
        ),
        blend=BlendConfig(
            dilate_px=int(bcfg.get("dilate_px", 0)),
            feather_px=int(bcfg.get("feather_px", 0)),
        ),
    )
    return Relocator(reloc_cfg)


# -----------------------------------------------------------------------------
# Core generation
# -----------------------------------------------------------------------------
def generate_same_background_scale(cfg: Dict[str, Any], overwrite: bool = False) -> None:
    exp_name = cfg["experiment"]["name"]
    val_root = cfg["paths"]["imagenet"]["val_root"]
    out_root = os.path.join(cfg["paths"]["outputs"]["root"], exp_name)
    out_images_dir = os.path.join(out_root, "images")
    out_meta_dir = os.path.join(out_root, "meta")
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)
    Path(out_meta_dir).mkdir(parents=True, exist_ok=True)

    content_manifest = cfg["content"]["manifest"]
    with open(content_manifest, "r") as f:
        base = json.load(f)

    scale_factors = [float(x) for x in (cfg.get("scales", {}) or {}).get("factors", [1.0, 0.75, 0.50, 0.25])]
    jpeg_quality = int((cfg.get("generation", {}) or {}).get("jpeg_quality", 95))
    inpaint_mask_dilate_px = int((cfg.get("pipeline", {}) or {}).get("inpaint_mask_dilate_px", 25))

    segmenter = build_segmenter()
    inpainter = build_inpainter()
    relocator = build_center_relocator(cfg)

    # Use the pipeline only as orchestrator for segmentation + inpainting.
    pipeline = AugmentationPipeline(
        segmenter=segmenter,
        inpainter=inpainter,
        relocator=None,
        postprocessor=None,
        save_intermediate=False,
        save_dir=None,
        log_pipeline=bool((cfg.get("pipeline", {}) or {}).get("log_pipeline", False)),
        inpaint_mask_dilate_px=inpaint_mask_dilate_px,
        apply_object_augmentation=False,
        enable_inpainting=True,
        enable_relocation=False,
    )

    print("[INFO] Generating RESCALE-SIZE FOR PREDICTION dataset")
    print("[INFO] Experiment:", exp_name)
    print("[INFO] Content samples:", len(base))
    print("[INFO] Scale factors:", scale_factors)
    print("[INFO] Output root:", out_root)

    gen_manifest: List[Dict[str, Any]] = []
    total_saved = 0

    for item in base:
        shape_wnid = item["wnid"]
        shape_name = item["class_name"]
        image_id = item["image_id"]
        content_rel = item["original_rel_path"]
        bbox = item["primary_bbox_xyxy"]

        content_path = os.path.join(val_root, content_rel)
        content = load_rgb(content_path)
        H, W = content.shape[:2]

        results = pipeline(
            image_rgb=content,
            boxes=[bbox],
            image_id=image_id,
            target_canvas=None,
        )
        if not results:
            print(f"[WARN] No pipeline result for {image_id}")
            continue
        result = results[0]
        if "error" in result:
            print(f"[WARN] Pipeline error for {image_id}: {result['error']}")
            continue

        object_rgb_full = result["object_rgb"]
        object_mask_full = to_u8_mask(result["original_mask"])
        inpainted_bg = result["inpainted_background"]

        obj_crop, mask_crop, crop_bbox_xyxy = crop_object_tight(object_rgb_full, object_mask_full)
        crop_h, crop_w = mask_crop.shape[:2]
        crop_area = int((mask_crop > 0).sum())

        for sf in scale_factors:
            obj_scaled, mask_scaled = resize_object_crop(obj_crop, mask_crop, sf)
            out_img, out_mask, reloc_meta = relocator.relocate(
                obj_rgb=obj_scaled,
                mask=mask_scaled,
                source_canvas=inpainted_bg,
                target_canvas=None,
                return_meta=True,
            )

            s_tag = scale_tag(sf)
            scale_pct = int(round(sf * 100.0))
            out_rel = os.path.join(
                "images",
                s_tag,
                shape_wnid,
                f"{image_id}__scale-{scale_pct:03d}.jpg",
            )
            out_path = os.path.join(out_root, out_rel)

            if overwrite or (not os.path.exists(out_path)):
                save_rgb(out_path, out_img, quality=jpeg_quality)
                total_saved += 1

            gen_manifest.append({
                "image_id": image_id,
                "shape_wnid": shape_wnid,
                "shape_name": shape_name,
                "content_rel_path": content_rel,
                "primary_bbox_xyxy": bbox,
                "image_size_hw": item.get("image_size_hw", [H, W]),
                "same_background": True,
                "background_type": "inpainted_original",
                "placement": "center",
                "scale_factor": float(sf),
                "scale_pct": scale_pct,
                "scale_tag": s_tag,
                "object_crop_bbox_xyxy": list(map(int, crop_bbox_xyxy)),
                "object_crop_hw": [int(crop_h), int(crop_w)],
                "object_crop_mask_area": crop_area,
                "scaled_object_hw": [int(obj_scaled.shape[0]), int(obj_scaled.shape[1])],
                "relocation_meta": reloc_meta,
                "output_rel_path": out_rel,
                "method": "rescale_size_for_prediction",
            })

    out_manifest_path = os.path.join(out_meta_dir, "rescale_size_for_prediction_gen_manifest.json")
    with open(out_manifest_path, "w") as f:
        json.dump(gen_manifest, f, indent=2)

    run_meta = {
        "experiment_name": exp_name,
        "num_content_images": len(base),
        "num_outputs_written": total_saved,
        "num_manifest_entries": len(gen_manifest),
        "scale_factors": scale_factors,
        "placement": "center",
        "same_background": True,
        "content_manifest": content_manifest,
    }
    with open(os.path.join(out_meta_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[INFO] Done. Saved {total_saved} images.")
    print("[INFO] Manifest:", out_manifest_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Experiment YAML")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    cfg = load_config_with_inherits(args.config)
    generate_same_background_scale(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
