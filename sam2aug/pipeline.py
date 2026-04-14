"""
Pipeline orchestration for the SAM2AUG framework.

This module defines the AugmentationPipeline, which coordinates the full
image processing workflow:

    segmentation → postprocessing → inpainting → augmentation → relocation

The pipeline itself does not implement model logic. Instead, it delegates to:
- segmenter.py        (SAM2-based mask generation)
- postprocessor.py    (object extraction and masking)
- inpainter.py        (LaMa-based inpainting)
- relocator.py        (object placement and blending)

Design principles:
- Modular: components can be enabled/disabled via flags
- Non-intrusive: does not modify model internals
- Reproducible: deterministic execution given inputs and config

"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

class AugmentationPipeline:
    """
    Orchestrator-only augmentation pipeline.

    Supported modes
    ---------------
    1) Segmentation only
       - useful for cutout / texture-replacement experiments

    2) Segmentation + inpainting
       - useful for generating clean backgrounds

    3) Segmentation + relocation
       - relocates onto:
           a) target_canvas if provided
           b) otherwise original image_rgb

    4) Full pipeline: segmentation + inpainting + relocation
       - relocates onto:
           a) target_canvas if provided
           b) otherwise inpainted background

    Responsibilities
    ----------------
    - call segmenter
    - extract object and source-with-hole
    - optionally inpaint
    - optionally augment object/mask before relocation
    - optionally relocate
    - collect outputs and metadata

    Notes
    -----
    - Segmentation, inpainting, and relocation remain modular.
    - This class is only the orchestrator.
    - Relocation logic belongs in relocator.py.
    """

    def __init__(
        self,
        segmenter: Any,
        inpainter: Optional[Any] = None,
        relocator: Optional[Any] = None,
        postprocessor: Optional[Any] = None,
        save_intermediate: bool = False,
        save_dir: Optional[str] = None,
        log_pipeline: bool = False,
        inpaint_mask_dilate_px: int = 25,
        apply_object_augmentation: bool = True,
        enable_inpainting: bool = True,
        enable_relocation: bool = True,
    ):
        self.segmenter = segmenter
        self.inpainter = inpainter
        self.relocator = relocator
        self.postprocessor = postprocessor

        self.save_intermediate = save_intermediate
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.log_pipeline = log_pipeline

        self.inpaint_mask_dilate_px = int(inpaint_mask_dilate_px)
        self.apply_object_augmentation = bool(apply_object_augmentation)

        self.enable_inpainting = bool(enable_inpainting)
        self.enable_relocation = bool(enable_relocation)

        if self.save_intermediate and self.save_dir is None:
            raise ValueError("save_dir must be provided when save_intermediate=True")

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.enable_inpainting and self.inpainter is None:
            raise ValueError("enable_inpainting=True but no inpainter was provided.")

        if self.enable_relocation and self.relocator is None:
            raise ValueError("enable_relocation=True but no relocator was provided.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(
        self,
        image_rgb: np.ndarray,
        boxes: Sequence[Sequence[float]],
        image_id: str = "unknown",
        target_canvas: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        return self.augment(
            image_rgb=image_rgb,
            boxes=boxes,
            image_id=image_id,
            target_canvas=target_canvas,
        )

    def augment(
        self,
        image_rgb: np.ndarray,
        boxes: Sequence[Sequence[float]],
        image_id: str = "unknown",
        target_canvas: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the configured pipeline for all provided boxes.

        Parameters
        ----------
        image_rgb:
            Source RGB image, HxWx3 uint8.
        boxes:
            Sequence of bounding boxes. Format must match the segmenter.
        image_id:
            Identifier for logging / saving.
        target_canvas:
            Optional external target RGB image for relocation.

        Returns
        -------
        List[Dict[str, Any]]
            One result dict per object / box.
        """
        self._validate_rgb(image_rgb, name="image_rgb")

        if target_canvas is not None:
            self._validate_rgb(target_canvas, name="target_canvas")

        if not boxes:
            return []

        t0_total = time.time()

        # 1) SEGMENTATION
        t0 = time.time()
        masks = self._segment(image_rgb=image_rgb, boxes=boxes)
        t_segment = time.time() - t0

        results: List[Dict[str, Any]] = []

        for idx, (box, mask) in enumerate(zip(boxes, masks)):
            object_id = f"{image_id}_{idx:03d}"
            t_item0 = time.time()

            try:
                mask_u8 = self._to_binary_mask(mask)

                # 2) POSTPROCESSING / EXTRACTION
                t0 = time.time()
                obj_rgb, source_with_hole = self._extract_object_and_source(
                    image_rgb=image_rgb,
                    mask_u8=mask_u8,
                )
                inpaint_mask = self._prepare_inpaint_mask(mask_u8) if self.enable_inpainting else None
                t_post = time.time() - t0

                # 3) OPTIONAL INPAINTING
                if self.enable_inpainting:
                    t0 = time.time()
                    inpainted_bg = self._inpaint(
                        image_rgb=source_with_hole,
                        mask_u8=inpaint_mask,
                    )
                    t_inpaint = time.time() - t0
                else:
                    inpainted_bg = None
                    t_inpaint = 0.0

                # 4) OPTIONAL OBJECT AUGMENTATION
                # Only really needed for relocation. If relocation is disabled,
                # keep original object/mask to support segmentation-only use cases.
                if self.enable_relocation and self.apply_object_augmentation:
                    t0 = time.time()
                    obj_for_reloc, mask_for_reloc = self._augment_object_and_mask(
                        obj_rgb=obj_rgb,
                        mask_u8=mask_u8,
                    )
                    t_obj_aug = time.time() - t0
                else:
                    obj_for_reloc, mask_for_reloc = obj_rgb, mask_u8
                    t_obj_aug = 0.0

                # 5) OPTIONAL RELOCATION
                if self.enable_relocation:
                    t0 = time.time()
                    relocation_canvas = self._resolve_relocation_canvas(
                        image_rgb=image_rgb,
                        inpainted_bg=inpainted_bg,
                        target_canvas=target_canvas,
                    )

                    relocated_img, relocated_mask, reloc_meta = self._relocate(
                        obj_rgb=obj_for_reloc,
                        mask_u8=mask_for_reloc,
                        source_canvas=relocation_canvas,
                    )
                    t_reloc = time.time() - t0
                else:
                    relocated_img = None
                    relocated_mask = None
                    reloc_meta = {}
                    relocation_canvas = None
                    t_reloc = 0.0

                item_time = time.time() - t_item0

                result = {
                    "image_id": image_id,
                    "object_id": object_id,
                    "object_index": idx,
                    "box": box,
                    "original_mask": mask_u8,
                    "inpaint_mask": inpaint_mask,
                    "object_rgb": obj_rgb,
                    "object_aug_rgb": obj_for_reloc if self.enable_relocation else None,
                    "object_aug_mask": mask_for_reloc if self.enable_relocation else None,
                    "source_with_hole": source_with_hole,
                    "inpainted_background": inpainted_bg,
                    "target_canvas": target_canvas,
                    "relocation_canvas": relocation_canvas,
                    "relocated_image": relocated_img,
                    "relocated_mask": relocated_mask,
                    "meta": {
                        "pipeline_flags": {
                            "enable_inpainting": self.enable_inpainting,
                            "enable_relocation": self.enable_relocation,
                            "apply_object_augmentation": self.apply_object_augmentation,
                        },
                        "timings": {
                            "segment_s": t_segment if idx == 0 else None,
                            "postprocess_s": t_post,
                            "inpaint_s": t_inpaint,
                            "object_augmentation_s": t_obj_aug,
                            "relocation_s": t_reloc,
                            "item_total_s": item_time,
                        },
                        "relocation": reloc_meta,
                    },
                }

                results.append(result)

                if self.save_intermediate:
                    self._save_result_bundle(result)

                if self.log_pipeline:
                    self._log_result(result)

            except Exception as e:
                error_result = {
                    "image_id": image_id,
                    "object_id": object_id,
                    "object_index": idx,
                    "box": box,
                    "error": str(e),
                    "meta": {
                        "pipeline_flags": {
                            "enable_inpainting": self.enable_inpainting,
                            "enable_relocation": self.enable_relocation,
                            "apply_object_augmentation": self.apply_object_augmentation,
                        }
                    },
                }
                results.append(error_result)

                if self.log_pipeline:
                    print(f"[Pipeline] ERROR for {object_id}: {e}")

        if self.log_pipeline:
            total_time = time.time() - t0_total
            print(
                f"[Pipeline] Completed {len(results)} objects for image_id={image_id} "
                f"in {total_time:.3f}s"
            )

        return results

    # ------------------------------------------------------------------
    # Stage 1: Segmentation
    # ------------------------------------------------------------------
    def _segment(
        self,
        image_rgb: np.ndarray,
        boxes: Sequence[Sequence[float]],
    ) -> List[np.ndarray]:
        """
        Accept multiple common segmenter interfaces and normalize to a list of HxW masks.
        Supports:
        - segment(...)
        - segment_image(...)
        - predict(...)
        - predict_masks(...)
        - callable(...)
        """
        raw = self.segmenter.segment_image(image_rgb, boxes)

        # Handle SAM2-style output: [(mask, score, box), ...]
        if isinstance(raw, (list, tuple)) and len(raw) > 0:
            first = raw[0]
            if isinstance(first, (list, tuple)) and len(first) >= 1:
                raw = [item[0] for item in raw]

        masks = self._normalize_masks(
            raw_masks=raw,
            expected_count=len(boxes),
            image_shape=image_rgb.shape[:2],
        )
        return masks

    # ------------------------------------------------------------------
    # Stage 2: Postprocessing / extraction
    # ------------------------------------------------------------------
    def _extract_object_and_source(
        self,
        image_rgb: np.ndarray,
        mask_u8: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prefer external postprocessor if available, otherwise use fallback:
          - object_rgb: source masked by object
          - source_with_hole: source with object area set to black
        """
        if self.postprocessor is not None:
            return self.postprocessor.extract_object_and_background(image_rgb, mask_u8)

        # Fallback
        obj_rgb = np.zeros_like(image_rgb, dtype=np.uint8)
        obj_rgb[mask_u8 > 0] = image_rgb[mask_u8 > 0]

        source_with_hole = image_rgb.copy()
        source_with_hole[mask_u8 > 0] = 0
        return obj_rgb, source_with_hole

    def _prepare_inpaint_mask(self, mask_u8: np.ndarray) -> np.ndarray:
        if self.inpaint_mask_dilate_px <= 0:
            return mask_u8.copy()

        k = 2 * self.inpaint_mask_dilate_px + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        dilated = (dilated > 0).astype(np.uint8) * 255
        return dilated

    # ------------------------------------------------------------------
    # Stage 3: Inpainting
    # ------------------------------------------------------------------
    def _inpaint(self, image_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        return self.inpainter.inpaint(image_rgb, mask_u8)

    # ------------------------------------------------------------------
    # Stage 4: Optional object augmentation
    # ------------------------------------------------------------------
    def _augment_object_and_mask(
        self,
        obj_rgb: np.ndarray,
        mask_u8: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(self.relocator, "augment_object_and_mask"):
            return self.relocator.augment_object_and_mask(obj_rgb, mask_u8)

        try:
            from relocator import augment_object_and_mask  # type: ignore
            return augment_object_and_mask(obj_rgb, mask_u8)
        except Exception:
            return obj_rgb, mask_u8

    # ------------------------------------------------------------------
    # Stage 5: Relocation
    # ------------------------------------------------------------------
    def _resolve_relocation_canvas(
        self,
        image_rgb: np.ndarray,
        inpainted_bg: Optional[np.ndarray],
        target_canvas: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Priority:
          1) explicit external target_canvas
          2) inpainted background (if available)
          3) original image
        """
        if target_canvas is not None:
            return target_canvas
        if self.enable_inpainting and inpainted_bg is not None:
            return inpainted_bg
        return image_rgb

    def _relocate(
        self,
        obj_rgb: np.ndarray,
        mask_u8: np.ndarray,
        source_canvas: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if hasattr(self.relocator, "relocate"):
            out = self.relocator.relocate(
                obj_rgb=obj_rgb,
                mask=mask_u8,
                source_canvas=source_canvas,
                target_canvas=None,
                return_meta=True,
            )

            if isinstance(out, tuple) and len(out) == 3:
                relocated_img, relocated_mask, reloc_meta = out
                return relocated_img, relocated_mask, reloc_meta

            if isinstance(out, tuple) and len(out) == 2:
                relocated_img, relocated_mask = out
                return relocated_img, relocated_mask, {}

        raise AttributeError("Relocator must provide relocate(...).")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_rgb(image_rgb: np.ndarray, name: str = "image_rgb") -> None:
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"{name} must be HxWx3 RGB image")
        if image_rgb.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8")

    @staticmethod
    def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
        if not isinstance(mask, np.ndarray):
            raise TypeError("Mask must be a numpy array")
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() == 1:
            mask = mask * 255
        mask = (mask > 0).astype(np.uint8) * 255
        return mask

    def _normalize_masks(
        self,
        raw_masks: Any,
        expected_count: int,
        image_shape: Tuple[int, int],
    ) -> List[np.ndarray]:
        """
        Normalize possible segmenter outputs to List[HxW uint8 mask].
        """
        h, w = image_shape

        if isinstance(raw_masks, (list, tuple)):
            masks = [self._to_binary_mask(m) for m in raw_masks]
            self._check_mask_count(masks, expected_count)
            self._check_mask_shapes(masks, h, w)
            return masks

        if isinstance(raw_masks, np.ndarray):
            if raw_masks.ndim == 2:
                masks = [self._to_binary_mask(raw_masks)]
                self._check_mask_count(masks, expected_count)
                self._check_mask_shapes(masks, h, w)
                return masks

            if raw_masks.ndim == 3:
                if raw_masks.shape[0] == expected_count and raw_masks.shape[1:] == (h, w):
                    masks = [self._to_binary_mask(raw_masks[i]) for i in range(raw_masks.shape[0])]
                    self._check_mask_shapes(masks, h, w)
                    return masks

                if raw_masks.shape[:2] == (h, w) and raw_masks.shape[2] == expected_count:
                    masks = [self._to_binary_mask(raw_masks[..., i]) for i in range(raw_masks.shape[2])]
                    self._check_mask_shapes(masks, h, w)
                    return masks

                if raw_masks.shape[0] == 1 and raw_masks.shape[1:] == (h, w):
                    masks = [self._to_binary_mask(raw_masks[0])]
                    self._check_mask_count(masks, expected_count)
                    self._check_mask_shapes(masks, h, w)
                    return masks

        raise ValueError(
            "Could not normalize segmenter output to a list of masks. "
            f"Expected {expected_count} masks for image shape {(h, w)}."
        )

    @staticmethod
    def _check_mask_count(masks: List[np.ndarray], expected_count: int) -> None:
        if len(masks) != expected_count:
            raise ValueError(
                f"Segmenter returned {len(masks)} masks, but {expected_count} boxes were provided."
            )

    @staticmethod
    def _check_mask_shapes(masks: List[np.ndarray], h: int, w: int) -> None:
        for i, m in enumerate(masks):
            if m.shape != (h, w):
                raise ValueError(f"Mask {i} has shape {m.shape}, expected {(h, w)}")

    # ------------------------------------------------------------------
    # Saving / logging
    # ------------------------------------------------------------------
    def _save_result_bundle(self, result: Dict[str, Any]) -> None:
        if self.save_dir is None:
            return

        obj_dir = self.save_dir / str(result["object_id"])
        obj_dir.mkdir(parents=True, exist_ok=True)

        save_map = {
            "01_original_mask.png": result.get("original_mask"),
            "02_inpaint_mask.png": result.get("inpaint_mask"),
            "03_object_rgb.png": result.get("object_rgb"),
            "04_object_aug_rgb.png": result.get("object_aug_rgb"),
            "05_object_aug_mask.png": result.get("object_aug_mask"),
            "06_source_with_hole.png": result.get("source_with_hole"),
            "07_inpainted_background.png": result.get("inpainted_background"),
            "08_relocated_image.png": result.get("relocated_image"),
            "09_relocated_mask.png": result.get("relocated_mask"),
        }

        target_canvas = result.get("target_canvas")
        if target_canvas is not None:
            save_map["00_target_canvas.png"] = target_canvas

        for filename, arr in save_map.items():
            if arr is None:
                continue
            self._save_image(obj_dir / filename, arr)

    @staticmethod
    def _save_image(path: Path, arr: np.ndarray) -> None:
        if arr.ndim == 2:
            cv2.imwrite(str(path), arr)
            return

        if arr.ndim == 3 and arr.shape[2] == 3:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), bgr)
            return

        raise ValueError(f"Cannot save array with shape {arr.shape} to {path}")

    def _log_result(self, result: Dict[str, Any]) -> None:
        obj_id = result.get("object_id", "unknown")

        if "error" in result:
            print(f"[Pipeline] object_id={obj_id}")
            print(f"  error={result['error']}")
            return

        meta = result.get("meta", {})
        flags = meta.get("pipeline_flags", {})
        timings = meta.get("timings", {})
        reloc = meta.get("relocation", {})

        print(f"[Pipeline] object_id={obj_id}")
        print(f"  enable_inpainting={flags.get('enable_inpainting')}")
        print(f"  enable_relocation={flags.get('enable_relocation')}")
        print(f"  apply_object_augmentation={flags.get('apply_object_augmentation')}")
        print(f"  postprocess_s={timings.get('postprocess_s')}")
        print(f"  inpaint_s={timings.get('inpaint_s')}")
        print(f"  object_augmentation_s={timings.get('object_augmentation_s')}")
        print(f"  relocation_s={timings.get('relocation_s')}")
        print(f"  item_total_s={timings.get('item_total_s')}")
        if reloc:
            print(f"  relocation_meta={reloc}")