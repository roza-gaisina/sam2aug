"""
Object relocation and blending module.

This module handles placing extracted objects onto a target canvas,
with support for:

    - placement strategies (center, random, etc.)
    - scaling
    - mask-based blending
    - optional mask dilation and feathering

Input:
    - object image (RGB)
    - object mask
    - source or target background

Output:
    - relocated image
    - relocated mask
    - optional metadata (placement info)

This module is responsible for all geometric and visual transformations
related to object placement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Existing augmentation helper (kept compatible with your current usage)
# -----------------------------------------------------------------------------
def augment_object_and_mask(obj_img: np.ndarray, mask_img: np.ndarray):
    """
    Apply identical random affine transform to object and mask.
    Expects:
      - obj_img: HxWx3 RGB uint8
      - mask_img: HxW uint8 {0,255} or {0,1}
    Returns:
      - obj_aug: HxWx3 RGB uint8
      - mask_aug: HxW uint8 {0,255}
    """
    h, w = obj_img.shape[:2]

    # Random rotation and scaling
    angle = random.uniform(-15, 15)
    scale = random.uniform(0.7, 1.3)

    # Random translation
    tx = random.uniform(-0.1, 0.1) * w
    ty = random.uniform(-0.1, 0.1) * h

    # Affine transform matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    obj_aug = cv2.warpAffine(
        obj_img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask_u8 = mask_img
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    if mask_u8.max() == 1:
        mask_u8 = mask_u8 * 255

    mask_aug = cv2.warpAffine(
        mask_u8,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    mask_aug = (mask_aug > 0).astype(np.uint8) * 255
    return obj_aug, mask_aug


# -----------------------------------------------------------------------------
# Relocation configs
# -----------------------------------------------------------------------------
PlacementMode = Literal["xy", "center", "random", "anchor", "region"]
AnchorName = Literal[
    "top_left",
    "top",
    "top_right",
    "left",
    "center",
    "right",
    "bottom_left",
    "bottom",
    "bottom_right",
]
ScaleMode = Literal["none", "area_fraction", "short_side", "bbox_fraction"]
RegionSampling = Literal["center", "random"]

# A region can be:
# - "grid cell": grid=(rows, cols), cell=(r, c)
# - "custom rect": rect=(x0,y0,x1,y1) normalized in [0,1]
@dataclass
class RegionConfig:
    grid: Optional[Tuple[int, int]] = None          # (rows, cols)
    cell: Optional[Tuple[int, int]] = None          # (row, col) 0-indexed
    rect: Optional[Tuple[float, float, float, float]] = None  # normalized [0,1]
    sampling: RegionSampling = "center"             # center or random


@dataclass
class PlacementConfig:
    mode: PlacementMode = "random"
    # for mode="xy"
    xy: Optional[Tuple[int, int]] = None
    # for mode="anchor"
    anchor: AnchorName = "center"
    # for mode="region"
    region: Optional[RegionConfig] = None
    # general constraints
    require_full_visibility: bool = True
    margin_px: int = 0
    max_attempts: int = 50
    # optional jitter applied to the final (x,y)
    jitter_px: int = 0


@dataclass
class ScaleConfig:
    mode: ScaleMode = "none"
    # area_fraction: object mask area ~ frac * target_area
    area_fraction: Optional[float] = None
    # short_side: set min(obj_h,obj_w)=short_side_px
    short_side_px: Optional[int] = None
    # bbox_fraction: set obj_w = w_frac*target_w and/or obj_h = h_frac*target_h
    bbox_fraction: Optional[Tuple[Optional[float], Optional[float]]] = None  # (w_frac, h_frac)
    # guardrails
    min_scale: float = 0.2
    max_scale: float = 3.0
    shrink_to_fit: bool = True  # if full visibility required and object doesn't fit


@dataclass
class BlendConfig:
    # Dilation applied to binary mask before feathering
    dilate_px: int = 0
    # Feather applied as Gaussian blur radius (in px) -> kernel size = 2*feather_px+1
    feather_px: int = 0
    # Threshold used to build placed_binary_mask output
    placed_mask_threshold: float = 0.5


@dataclass
class RelocatorConfig:
    placement: PlacementConfig = PlacementConfig()
    scale: ScaleConfig = ScaleConfig()
    blend: BlendConfig = BlendConfig()


# -----------------------------------------------------------------------------
# Relocator implementation
# -----------------------------------------------------------------------------
class Relocator:
    """
    Flexible object relocation engine.

    Pipeline steps:
        1. Normalize mask
        2. Optionally scale object
        3. Generate alpha map (dilation + feathering)
        4. Choose placement position
        5. Composite object onto target canvas

    Supports:
        - Multiple placement strategies (center, random, anchor, region)
        - Flexible scaling policies
        - Alpha blending with mask preprocessing

    Input assumptions:
        - obj_rgb: np.ndarray (H, W, 3), RGB uint8
        - mask: np.ndarray (H, W), binary {0,1} or {0,255}
        - canvas: np.ndarray (H, W, 3), RGB uint8
    """

    def __init__(self, cfg: Optional[RelocatorConfig] = None):
        self.cfg = cfg or RelocatorConfig()

    # -----------------------------
    # Mask preprocessing -> alpha
    # -----------------------------
    @staticmethod
    def _to_u8_binary_mask(mask: np.ndarray) -> np.ndarray:
        """
        Converts mask to binary uint8 format {0,255}.

        Accepts:
            - {0,1}
            - {0,255}
            - any non-zero values
        """
        m = mask
        if m is None:
            raise ValueError("Mask is None")
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)
        if m.ndim == 3:
            # If somehow provided as HxWx1 or HxWx3, reduce to 1 channel
            m = m[..., 0]
        if m.max() == 1:
            m = m * 255
        m = (m > 0).astype(np.uint8) * 255
        return m

    def preprocess_alpha(self, mask_u8: np.ndarray) -> np.ndarray:
        """
        Generates alpha map in [0,1] from binary mask.

        Steps:
            1. Optional dilation (expands object)
            2. Optional Gaussian blur (soft edges)
        """
        m = self._to_u8_binary_mask(mask_u8)

        if self.cfg.blend.dilate_px > 0:
            k = 2 * int(self.cfg.blend.dilate_px) + 1
            kernel = np.ones((k, k), np.uint8)
            m = cv2.dilate(m, kernel, iterations=1)

        if self.cfg.blend.feather_px > 0:
            k = 2 * int(self.cfg.blend.feather_px) + 1
            # Gaussian blur produces soft edges
            m = cv2.GaussianBlur(m, (k, k), 0)

        alpha = m.astype(np.float32) / 255.0
        alpha = np.clip(alpha, 0.0, 1.0)
        return alpha

    # -----------------------------
    # Scaling
    # -----------------------------
    def compute_scale(
        self,
        obj_rgb: np.ndarray,
        mask_u8: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> float:
        scfg = self.cfg.scale
        if scfg.mode == "none":
            return 1.0

        m = self._to_u8_binary_mask(mask_u8)
        obj_area = float((m > 0).sum())
        if obj_area <= 1:
            return 1.0

        if scfg.mode == "area_fraction":
            if not scfg.area_fraction:
                return 1.0
            target_area = float(target_h * target_w)
            desired_area = float(scfg.area_fraction) * target_area
            scale = float(np.sqrt(desired_area / obj_area))

        elif scfg.mode == "short_side":
            if not scfg.short_side_px:
                return 1.0
            oh, ow = obj_rgb.shape[:2]
            short_side = float(min(oh, ow))
            scale = float(scfg.short_side_px) / max(short_side, 1.0)

        elif scfg.mode == "bbox_fraction":
            if not scfg.bbox_fraction:
                return 1.0
            w_frac, h_frac = scfg.bbox_fraction
            oh, ow = obj_rgb.shape[:2]
            candidates = []
            if w_frac is not None:
                candidates.append((float(w_frac) * float(target_w)) / max(float(ow), 1.0))
            if h_frac is not None:
                candidates.append((float(h_frac) * float(target_h)) / max(float(oh), 1.0))
            scale = min(candidates) if candidates else 1.0

        else:
            scale = 1.0

        scale = float(np.clip(scale, scfg.min_scale, scfg.max_scale))
        return scale

    @staticmethod
    def resize_obj_and_mask(
        obj_rgb: np.ndarray, mask_u8: np.ndarray, scale: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        if abs(scale - 1.0) < 1e-6:
            return obj_rgb, mask_u8
        oh, ow = obj_rgb.shape[:2]
        new_w = max(1, int(round(ow * scale)))
        new_h = max(1, int(round(oh * scale)))

        obj_r = cv2.resize(obj_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask_r = (mask_r > 0).astype(np.uint8) * 255
        return obj_r, mask_r

    # -----------------------------
    # Placement helpers
    # -----------------------------
    @staticmethod
    def _clamp_int(x: int) -> int:
        return int(x)

    @staticmethod
    def _anchor_to_top_left(
        target_w: int,
        target_h: int,
        obj_w: int,
        obj_h: int,
        anchor: AnchorName,
        margin: int,
    ) -> Tuple[int, int]:
        """
        Deterministic top-left placement from an anchor command, with margin.
        Margin is interpreted as "keep object bbox away from edges by margin", when possible.
        """
        # base coordinates (before margin)
        if anchor in ("top_left",):
            x, y = 0, 0
        elif anchor in ("top",):
            x, y = (target_w - obj_w) // 2, 0
        elif anchor in ("top_right",):
            x, y = target_w - obj_w, 0
        elif anchor in ("left",):
            x, y = 0, (target_h - obj_h) // 2
        elif anchor in ("center",):
            x, y = (target_w - obj_w) // 2, (target_h - obj_h) // 2
        elif anchor in ("right",):
            x, y = target_w - obj_w, (target_h - obj_h) // 2
        elif anchor in ("bottom_left",):
            x, y = 0, target_h - obj_h
        elif anchor in ("bottom",):
            x, y = (target_w - obj_w) // 2, target_h - obj_h
        elif anchor in ("bottom_right",):
            x, y = target_w - obj_w, target_h - obj_h
        else:
            x, y = (target_w - obj_w) // 2, (target_h - obj_h) // 2

        # apply margin by nudging inward
        x = x + margin if x <= 0 else x - margin if x >= target_w - obj_w else x
        y = y + margin if y <= 0 else y - margin if y >= target_h - obj_h else y
        return int(x), int(y)

    @staticmethod
    def _region_rect_px(
        target_w: int,
        target_h: int,
        region: RegionConfig,
    ) -> Tuple[int, int, int, int]:
        """
        Returns region rectangle in pixel coords as (x0,y0,x1,y1), inclusive-exclusive.
        Supports grid cell and normalized rect.
        Defaults to full canvas if region is not properly specified.
        """
        # custom normalized rect
        if region.rect is not None:
            x0n, y0n, x1n, y1n = region.rect
            x0 = int(round(np.clip(x0n, 0.0, 1.0) * target_w))
            y0 = int(round(np.clip(y0n, 0.0, 1.0) * target_h))
            x1 = int(round(np.clip(x1n, 0.0, 1.0) * target_w))
            y1 = int(round(np.clip(y1n, 0.0, 1.0) * target_h))
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            x1 = max(x1, x0 + 1)
            y1 = max(y1, y0 + 1)
            return x0, y0, x1, y1

        # grid cell
        if region.grid is not None and region.cell is not None:
            rows, cols = region.grid
            r, c = region.cell
            rows = max(1, int(rows))
            cols = max(1, int(cols))
            r = int(np.clip(r, 0, rows - 1))
            c = int(np.clip(c, 0, cols - 1))

            cell_w = target_w / float(cols)
            cell_h = target_h / float(rows)

            x0 = int(round(c * cell_w))
            x1 = int(round((c + 1) * cell_w))
            y0 = int(round(r * cell_h))
            y1 = int(round((r + 1) * cell_h))

            x1 = max(x1, x0 + 1)
            y1 = max(y1, y0 + 1)
            return x0, y0, x1, y1

        # default: full canvas
        return 0, 0, target_w, target_h

    def _apply_jitter(
        self,
        x: int,
        y: int,
        jitter_px: int,
        rng: Union[random.Random, Any],
    ) -> Tuple[int, int]:
        if jitter_px <= 0:
            return x, y
        jx = int(rng.randint(-jitter_px, jitter_px))
        jy = int(rng.randint(-jitter_px, jitter_px))
        return int(x + jx), int(y + jy)

    def choose_position(
        self,
        target_h: int,
        target_w: int,
        obj_h: int,
        obj_w: int,
        rng: Optional[Union[random.Random, Any]] = None,
        placement_xy: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Returns top-left (x, y) position in target canvas coordinates.

        Coordinate system:
            (0,0) = top-left corner of canvas
        """
        pcfg = self.cfg.placement
        rng = rng or random
        margin = max(0, int(pcfg.margin_px))
        jitter = max(0, int(pcfg.jitter_px))

        # override mode if explicit xy is provided
        mode = pcfg.mode
        if placement_xy is not None:
            mode = "xy"

        # Helper: sample random top-left under constraints
        def sample_random_top_left(require_full: bool) -> Optional[Tuple[int, int]]:
            if require_full:
                if obj_w + 2 * margin > target_w or obj_h + 2 * margin > target_h:
                    return None
                x_min, x_max = margin, target_w - obj_w - margin
                y_min, y_max = margin, target_h - obj_h - margin
            else:
                # allow partial off-canvas placement
                x_min, x_max = -obj_w + margin, target_w - margin
                y_min, y_max = -obj_h + margin, target_h - margin

            if x_max < x_min or y_max < y_min:
                return None
            x = int(rng.randint(int(x_min), int(x_max))) if x_max > x_min else int(x_min)
            y = int(rng.randint(int(y_min), int(y_max))) if y_max > y_min else int(y_min)
            x, y = self._apply_jitter(x, y, jitter, rng)
            return x, y

        if mode == "center":
            x = (target_w - obj_w) // 2
            y = (target_h - obj_h) // 2
            x, y = self._apply_jitter(int(x), int(y), jitter, rng)
            return int(x), int(y)

        if mode == "xy":
            xy = placement_xy or pcfg.xy
            if xy is None:
                return None
            x, y = int(xy[0]), int(xy[1])
            x, y = self._apply_jitter(x, y, jitter, rng)
            return int(x), int(y)

        if mode == "anchor":
            x, y = self._anchor_to_top_left(
                target_w=target_w,
                target_h=target_h,
                obj_w=obj_w,
                obj_h=obj_h,
                anchor=pcfg.anchor,
                margin=margin,
            )
            x, y = self._apply_jitter(int(x), int(y), jitter, rng)
            return int(x), int(y)

        if mode == "region":
            region = pcfg.region or RegionConfig(grid=(2, 2), cell=(0, 0), sampling="center")
            x0, y0, x1, y1 = self._region_rect_px(target_w, target_h, region)

            # We place the object so that its *top-left* lies in a feasible sub-rectangle
            # that keeps the object inside (if full visibility required).
            if pcfg.require_full_visibility:
                fx0 = x0 + margin
                fy0 = y0 + margin
                fx1 = x1 - obj_w - margin
                fy1 = y1 - obj_h - margin
            else:
                # allow partial; still try to bias inside region
                fx0 = x0 - obj_w + margin
                fy0 = y0 - obj_h + margin
                fx1 = x1 - margin
                fy1 = y1 - margin

            if fx1 < fx0 or fy1 < fy0:
                return None

            if region.sampling == "center":
                x = int((fx0 + fx1) // 2)
                y = int((fy0 + fy1) // 2)
            else:
                x = int(rng.randint(int(fx0), int(fx1))) if fx1 > fx0 else int(fx0)
                y = int(rng.randint(int(fy0), int(fy1))) if fy1 > fy0 else int(fy0)

            x, y = self._apply_jitter(int(x), int(y), jitter, rng)
            return int(x), int(y)

        # mode == "random" (default)
        return sample_random_top_left(require_full=pcfg.require_full_visibility)

    # -----------------------------
    # Alpha compositing
    # -----------------------------
    def composite_alpha(
        self,
        target_rgb: np.ndarray,
        obj_rgb: np.ndarray,
        alpha: np.ndarray,  # [0,1] same HxW as obj
        top_left: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Composites obj onto target using alpha.
        Returns:
          - composited_rgb (HxWx3 uint8)
          - placed_binary_mask (HxW uint8 {0,255} in target coords)
        """
        out = target_rgb.copy()
        th, tw = out.shape[:2]
        oh, ow = obj_rgb.shape[:2]
        x0, y0 = int(top_left[0]), int(top_left[1])

        # target bounds
        x1 = max(0, x0)
        y1 = max(0, y0)
        x2 = min(tw, x0 + ow)
        y2 = min(th, y0 + oh)
        if x2 <= x1 or y2 <= y1:
            return out, np.zeros((th, tw), dtype=np.uint8)

        # object bounds
        ox1 = max(0, -x0)
        oy1 = max(0, -y0)
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        obj_crop = obj_rgb[oy1:oy2, ox1:ox2].astype(np.float32)
        a_crop = alpha[oy1:oy2, ox1:ox2].astype(np.float32)[..., None]
        tgt_crop = out[y1:y2, x1:x2].astype(np.float32)

        comp = obj_crop * a_crop + tgt_crop * (1.0 - a_crop)
        out[y1:y2, x1:x2] = np.clip(comp, 0, 255).astype(np.uint8)

        placed_mask = np.zeros((th, tw), dtype=np.uint8)
        thr = float(self.cfg.blend.placed_mask_threshold)
        placed_mask[y1:y2, x1:x2] = (a_crop[..., 0] > thr).astype(np.uint8) * 255
        return out, placed_mask

    # -----------------------------
    # Main entry point
    # -----------------------------
    def relocate(
        self,
        obj_rgb: np.ndarray,
        mask: np.ndarray,
        source_canvas: Optional[np.ndarray] = None,
        target_canvas: Optional[np.ndarray] = None,
        placement_xy: Optional[Tuple[int, int]] = None,
        rng: Optional[Union[random.Random, Any]] = None,
        return_meta: bool = False,
    ):
        """
        Relocate (obj_rgb, mask) onto target_canvas (if provided) else source_canvas.

        Inputs:
          - obj_rgb: HxWx3 RGB uint8
          - mask: HxW uint8 {0,255} or {0,1}
          - source_canvas: fallback background (e.g. inpainted background)
          - target_canvas: optional external background/canvas
          - placement_xy: optional override for xy placement
          - rng: optional random generator (random.Random or np.random.Generator-like)
          - return_meta: if True returns (img, mask, meta)

        Returns:
          - relocated_img: target-sized RGB uint8
          - relocated_mask: target-sized mask uint8 {0,255}
          - meta (optional): scale, pos, attempts, mode
        """
        canvas = target_canvas if target_canvas is not None else source_canvas
        if canvas is None:
            raise ValueError("Relocator.relocate: either source_canvas or target_canvas must be provided")

        if obj_rgb is None or mask is None:
            th, tw = canvas.shape[:2]
            empty = np.zeros((th, tw), dtype=np.uint8)
            return (canvas.copy(), empty, {"reason": "missing_obj_or_mask"}) if return_meta else (canvas.copy(), empty)

        if canvas.ndim != 3 or canvas.shape[2] != 3:
            raise ValueError("Relocator.relocate: canvas must be HxWx3 RGB image")

        th, tw = canvas.shape[:2]
        rng = rng or random
        
        # 1) Normalize mask
        mask_u8 = self._to_u8_binary_mask(mask)
        
        # 2) Compute scale
        scale = self.compute_scale(obj_rgb, mask_u8, th, tw)
        obj_s, mask_s = self.resize_obj_and_mask(obj_rgb, mask_u8, scale)
        oh, ow = obj_s.shape[:2]

        # 2b) Shrink-to-fit if required
        if self.cfg.placement.require_full_visibility and (ow > tw or oh > th):
            if self.cfg.scale.shrink_to_fit:
                fit_scale = min(tw / max(ow, 1), th / max(oh, 1), 1.0)
                # 3) Resize object and mask
                obj_s, mask_s = self.resize_obj_and_mask(obj_s, mask_s, fit_scale)
                oh, ow = obj_s.shape[:2]
                scale = scale * fit_scale
            else:
                empty = np.zeros((th, tw), dtype=np.uint8)
                meta = {"reason": "object_does_not_fit", "scale": scale}
                return (canvas.copy(), empty, meta) if return_meta else (canvas.copy(), empty)

        # 4) Build alpha map
        alpha = self.preprocess_alpha(mask_s)

        # 5) Try placement
        attempts = max(1, int(self.cfg.placement.max_attempts))

        for i in range(attempts):
            pos = self.choose_position(
                target_h=th,
                target_w=tw,
                obj_h=oh,
                obj_w=ow,
                rng=rng,
                placement_xy=placement_xy,
            )
            if pos is None:
                break

            x, y = pos
            if self.cfg.placement.require_full_visibility:
                # enforce bbox inside (including margin already handled in sampling)
                if x < 0 or y < 0 or (x + ow) > tw or (y + oh) > th:
                    continue

            out, placed = self.composite_alpha(canvas, obj_s, alpha, pos)

            meta = {
                "scale": float(scale),
                "pos": (int(x), int(y)),
                "attempt": int(i + 1),
                "placement_mode": self.cfg.placement.mode,
            }

            # For now, return the first valid placement
            if return_meta:
                return out, placed, meta
            return out, placed

            # (If you ever want a "best scoring" strategy, keep best here.)
            # best_out, best_mask, best_meta = out, placed, meta

        # fallback if no valid placement found
        empty = np.zeros((th, tw), dtype=np.uint8)
        meta = {
            "reason": "no_valid_placement",
            "scale": float(scale),
            "attempts": int(attempts),
            "placement_mode": self.cfg.placement.mode,
        }
        return (canvas.copy(), empty, meta) if return_meta else (canvas.copy(), empty)


# -----------------------------------------------------------------------------
# Backwards-compatible wrapper (optional)
# -----------------------------------------------------------------------------
def paste_with_visibility(
    background: np.ndarray,
    obj: np.ndarray,
    mask: np.ndarray,
    visibility_ratio: float = 1.0,
    max_attempts: int = 2,
):
    """
    Compatibility function matching your existing pipeline usage.

    NOTE:
      - It does NOT actively optimize for visibility_ratio anymore.
      - It approximates old behavior:
          visibility_ratio >= 0.99 -> require full visibility
          otherwise -> allow partial visibility (object can be partially off-canvas)
    """
    require_full = bool(visibility_ratio >= 0.99)

    cfg = RelocatorConfig(
        placement=PlacementConfig(
            mode="random",
            require_full_visibility=require_full,
            margin_px=0,
            max_attempts=max_attempts,
            jitter_px=0,
        ),
        scale=ScaleConfig(mode="none"),
        blend=BlendConfig(dilate_px=0, feather_px=0),
    )
    reloc = Relocator(cfg)
    return reloc.relocate(obj_rgb=obj, mask=mask, source_canvas=background)