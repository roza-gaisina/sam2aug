#!/usr/bin/env python3
"""
Generate Geirhos16 water-background relocation samples.

This script uses the modular augmentation pipeline with:
  - segmentation enabled
  - inpainting disabled
  - relocation enabled

Objects from the Geirhos16 ImageNet validation subset are segmented from their
source images and relocated onto one of three water backgrounds:
  - shore.jpg
  - openwater.jpg
  - surface.jpg

Outputs:
  <paths.outputs.root>/<experiment.name>/
    images/<background_name>/<wnid>/<image_id>__bg-<background>.jpg
    meta/water_relocation_manifest.json

Example:
  python generate_object_relocation_background_shift.py --config configs/object_relocation_background_shift.yaml --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
import numpy as np
from PIL import Image

from sam2aug.segmenter import Segmenter
from sam2aug.pipeline import AugmentationPipeline
from sam2aug.relocator import (
    Relocator,
    RelocatorConfig,
    PlacementConfig,
    ScaleConfig,
    BlendConfig,
)
import sam2aug.config as sam2cfg

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


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))



def save_rgb(path: str, rgb: np.ndarray, quality: int = 95) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=quality)


# -----------------------------------------------------------------------------
# Build pipeline
# -----------------------------------------------------------------------------
def build_pipeline(cfg: Dict[str, Any]) -> AugmentationPipeline:
    relocation_cfg = cfg.get("relocation", {})
    generation_cfg = cfg.get("generation", {})

    segmenter = Segmenter(
        model_config=sam2cfg.SAM2_CONFIG,
        checkpoint_path=sam2cfg.SAM2_CHECKPOINT,
        device=sam2cfg.DEVICE,
    )

    placement = PlacementConfig(
        mode=relocation_cfg.get("placement_mode", "center"),
        anchor=relocation_cfg.get("anchor", "center"),
        xy=tuple(relocation_cfg["xy"]) if relocation_cfg.get("xy") is not None else None,
        require_full_visibility=bool(relocation_cfg.get("require_full_visibility", True)),
        margin_px=int(relocation_cfg.get("margin_px", 8)),
        max_attempts=int(relocation_cfg.get("max_attempts", 20)),
        jitter_px=int(relocation_cfg.get("jitter_px", 0)),
    )

    scale = ScaleConfig(
        mode=relocation_cfg.get("scale_mode", "area_fraction"),
        area_fraction=float(relocation_cfg.get("area_fraction", 0.10))
        if relocation_cfg.get("area_fraction") is not None
        else None,
        short_side_px=relocation_cfg.get("short_side_px"),
        bbox_fraction=tuple(relocation_cfg["bbox_fraction"])
        if relocation_cfg.get("bbox_fraction") is not None
        else None,
        min_scale=float(relocation_cfg.get("min_scale", 0.25)),
        max_scale=float(relocation_cfg.get("max_scale", 1.8)),
        shrink_to_fit=bool(relocation_cfg.get("shrink_to_fit", True)),
    )

    blend = BlendConfig(
        dilate_px=int(generation_cfg.get("dilate_px", 2)),
        feather_px=int(generation_cfg.get("feather_px", 2)),
    )

    relocator = Relocator(RelocatorConfig(placement=placement, scale=scale, blend=blend))

    pipeline = AugmentationPipeline(
        segmenter=segmenter,
        inpainter=None,
        relocator=relocator,
        postprocessor=None,
        save_intermediate=False,
        save_dir=None,
        log_pipeline=bool(cfg.get("logging", {}).get("enabled", False)),
        inpaint_mask_dilate_px=0,
        apply_object_augmentation=bool(relocation_cfg.get("apply_object_augmentation", False)),
        enable_inpainting=False,
        enable_relocation=True,
    )
    return pipeline


# -----------------------------------------------------------------------------
# Main generation routine
# -----------------------------------------------------------------------------
def generate_background_shift(cfg: Dict[str, Any], overwrite: bool = False) -> None:
    val_root = cfg["paths"]["imagenet"]["val_root"]
    out_root = os.path.join(cfg["paths"]["outputs"]["root"], cfg["experiment"]["name"])
    manifest_path = cfg["content"]["manifest"]
    jpeg_quality = int(cfg.get("generation", {}).get("jpeg_quality", 95))

    backgrounds_dir = cfg["backgrounds"]["dir"]
    background_names: List[str] = list(cfg["backgrounds"]["names"])

    with open(manifest_path, "r") as f:
        base_manifest = json.load(f)

    pipeline = build_pipeline(cfg)

    background_cache: Dict[str, np.ndarray] = {}
    for name in background_names:
        bg_path = os.path.join(backgrounds_dir, f"{name}.jpg")
        if not os.path.exists(bg_path):
            raise FileNotFoundError(f"Background not found: {bg_path}")
        background_cache[name] = load_rgb(bg_path)

    out_images_dir = os.path.join(out_root, "images")
    out_meta_dir = os.path.join(out_root, "meta")
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)
    Path(out_meta_dir).mkdir(parents=True, exist_ok=True)

    generated_manifest: List[Dict[str, Any]] = []
    written = 0

    print("[INFO] Generating object relocation background shift dataset")
    print(f"[INFO] Experiment: {cfg['experiment']['name']}")
    print(f"[INFO] Content samples: {len(base_manifest)}")
    print(f"[INFO] Backgrounds: {background_names}")
    print(f"[INFO] Output root: {out_root}")

    for item in base_manifest:
        image_id = item["image_id"]
        wnid = item["wnid"]
        class_name = item["class_name"]
        content_rel = item["original_rel_path"]
        bbox = item["primary_bbox_xyxy"]

        image_path = os.path.join(val_root, content_rel)
        if not os.path.exists(image_path):
            print(f"[WARN] Missing source image: {image_path}")
            continue

        image_rgb = load_rgb(image_path)

        for bg_name in background_names:
            bg_rgb = background_cache[bg_name]

            results = pipeline(
                image_rgb=image_rgb,
                boxes=[bbox],
                image_id=image_id,
                target_canvas=bg_rgb,
            )

            if not results:
                print(f"[WARN] No result for {image_id} on {bg_name}")
                continue

            result = results[0]
            if "error" in result:
                print(f"[WARN] Failed for {image_id} on {bg_name}: {result['error']}")
                continue

            relocated = result["relocated_image"]
            relocated_mask = result["relocated_mask"]
            obj_rgb = result["object_rgb"]
            obj_mask = result["original_mask"]
            reloc_meta = result.get("meta", {}).get("relocation", {})

            out_rel = os.path.join(
                "images",
                bg_name,
                wnid,
                f"{image_id}__bg-{bg_name}.jpg",
            )
            out_path = os.path.join(out_root, out_rel)

            if overwrite or (not os.path.exists(out_path)):
                save_rgb(out_path, relocated, quality=jpeg_quality)
                written += 1

            generated_manifest.append(
                {
                    "image_id": image_id,
                    "wnid": wnid,
                    "class_name": class_name,
                    "content_rel_path": content_rel,
                    "primary_bbox_xyxy": bbox,
                    "source_image_size_hw": item.get("image_size_hw", list(image_rgb.shape[:2])),
                    "background_name": bg_name,
                    "background_rel_path": os.path.join(cfg["backgrounds"]["dir"], f"{bg_name}.jpg"),
                    "output_rel_path": out_rel,
                    "method": "background_shift",
                    "pipeline_flags": {
                        "enable_inpainting": False,
                        "enable_relocation": True,
                        "apply_object_augmentation": bool(
                            cfg.get("relocation", {}).get("apply_object_augmentation", False)
                        ),
                    },
                    "generation": {
                        "feather_px": int(cfg.get("generation", {}).get("feather_px", 2)),
                        "dilate_px": int(cfg.get("generation", {}).get("dilate_px", 2)),
                        "jpeg_quality": jpeg_quality,
                    },
                    "relocation": {
                        "placement_mode": cfg.get("relocation", {}).get("placement_mode", "center"),
                        "scale_mode": cfg.get("relocation", {}).get("scale_mode", "area_fraction"),
                        "area_fraction": cfg.get("relocation", {}).get("area_fraction", 0.10),
                        "margin_px": cfg.get("relocation", {}).get("margin_px", 8),
                        "meta": reloc_meta,
                    },
                    "artifacts": {
                        "object_mask_area_px": int((obj_mask > 0).sum()) if obj_mask is not None else None,
                        "relocated_mask_area_px": int((relocated_mask > 0).sum()) if relocated_mask is not None else None,
                        "object_rgb_shape": list(obj_rgb.shape) if obj_rgb is not None else None,
                        "output_rgb_shape": list(relocated.shape) if relocated is not None else None,
                    },
                }
            )

    manifest_out = os.path.join(out_meta_dir, "background_shift_manifest.json")
    with open(manifest_out, "w") as f:
        json.dump(generated_manifest, f, indent=2)

    print(f"[INFO] Done. Wrote {written} images.")
    print(f"[INFO] Manifest: {manifest_out}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config_with_inherits(args.config)
    generate_background_shift(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
