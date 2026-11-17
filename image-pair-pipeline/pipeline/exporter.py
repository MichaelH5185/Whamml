"""
exporter.py
-----------
Writes the two output repositories:

1. output_crops/class{N}/
   - Padded bounding-box crop from BEFORE image  (JPEG)
   - Padded bounding-box crop from AFTER  image  (JPEG)

2. output_masks/class{N}/
   - Segmentation mask crop from BEFORE image, padded to square (PNG)
   - Segmentation mask crop from AFTER  image, padded to square (PNG)

Each image is cropped using its OWN detection (before-det → before-image,
after-det → after-image).  No cross-application of masks.

Filename pattern
    {suffix}_{pair_idx:02d}_before.jpg / _after.jpg
    {suffix}_{pair_idx:02d}_before.png / _after.png
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from .detector import Detection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _padded_bbox(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
    pad: float = 0.20,
) -> tuple[int, int, int, int]:
    """Expand a bbox by *pad* fraction on each side, clamped to image bounds."""
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad
    py = bh * pad
    cx1 = max(0,     int(x1 - px))
    cy1 = max(0,     int(y1 - py))
    cx2 = min(img_w, int(x2 + px))
    cy2 = min(img_h, int(y2 + py))
    return cx1, cy1, cx2, cy2


def _pad_to_square(img: np.ndarray, pad_color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Pad *img* to a square with *pad_color* (no resize, just black bars)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    if len(img.shape) == 2:
        canvas = np.full((size, size), pad_color[0], dtype=img.dtype)
    else:
        canvas = np.full((size, size, img.shape[2]), pad_color, dtype=img.dtype)
    # Centre the image
    top  = (size - h) // 2
    left = (size - w) // 2
    canvas[top:top + h, left:left + w] = img
    return canvas


def _crop_masked_object(
    img: np.ndarray,
    mask_poly: Optional[np.ndarray],
    bbox: list[float],
) -> np.ndarray:
    """
    Apply the segmentation mask directly to the image (keeping building pixels,
    zeroing background pixels to black), crop tightly to the bounding box / mask extents,
    and pad with black pixels to make it square.
    """
    img_h, img_w = img.shape[:2]
    mask_canvas = np.zeros((img_h, img_w), dtype=np.uint8)

    if mask_poly is not None and len(mask_poly) >= 3:
        pts = mask_poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask_canvas, [pts], 255)
        # Tight crop to mask bounding rect or detection bbox
        bx, by, bw, bh = cv2.boundingRect(pts)
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(img_w, bx + bw), min(img_h, by + bh)
    else:
        # Fall back to detection bbox
        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        x1, y1 = max(0, bx1), max(0, by1)
        x2, y2 = min(img_w, bx2), min(img_h, by2)
        cv2.rectangle(mask_canvas, (x1, y1), (x2, y2), 255, -1)

    # Apply mask to image (keep 3-channel RGB image pixels where mask == 255, else black)
    masked_img = cv2.bitwise_and(img, img, mask=mask_canvas)

    # Crop tightly to the object
    crop = masked_img[y1:y2, x1:x2]
    return _pad_to_square(crop)


def _render_mask(
    mask_poly: Optional[np.ndarray],
    bbox: list[float],
    img_h: int,
    img_w: int,
    cx1: int, cy1: int, cx2: int, cy2: int,
) -> np.ndarray:
    """
    Render segmentation polygon as white on a black canvas (full image size),
    then crop to the padded bbox region and pad to square.
    (Kept for backward compatibility).
    """
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)

    if mask_poly is not None and len(mask_poly) >= 3:
        pts = mask_poly.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], 255)
    else:
        # Fall back: fill bbox as mask
        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), 255, -1)

    crop = canvas[cy1:cy2, cx1:cx2]
    return _pad_to_square(crop)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Exporter:
    def __init__(
        self,
        crops_root: str | Path,
        masks_root: str | Path,
        crop_pad: float = 0.20,
        jpeg_quality: int = 95,
    ):
        self.crops_root = Path(crops_root)
        self.masks_root = Path(masks_root)
        self.crop_pad = crop_pad
        self.jpeg_quality = jpeg_quality
        # Per-building pair counter to generate sequential indices
        self._counters: dict[tuple, int] = {}

    def _next_idx(self, building_key: tuple) -> int:
        idx = self._counters.get(building_key, 0)
        self._counters[building_key] = idx + 1
        return idx

    def export_pair(
        self,
        suffix: str,
        damage_class: int,
        before_path: Path,
        after_path: Path,
        det_before: Detection,
        det_after: Detection,
        building_key: Optional[tuple] = None,
    ) -> dict[str, Path]:
        """
        Write one pair of crops and one pair of masks.

        Returns
        -------
        Dict with keys "crop_before", "crop_after", "mask_before", "mask_after".
        """
        if building_key is None:
            building_key = (suffix,)

        pair_idx = self._next_idx(building_key)
        cls_name = f"class{damage_class}"
        stem = f"{suffix}_{pair_idx:02d}"

        crops_dir = self.crops_root / cls_name
        masks_dir = self.masks_root / cls_name
        crops_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        saved: dict[str, Path] = {}

        for role, img_path, det in [
            ("before", before_path, det_before),
            ("after",  after_path,  det_after),
        ]:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Cannot read image: {img_path}")
            img_h, img_w = img.shape[:2]

            x1, y1, x2, y2 = det.bbox
            cx1, cy1, cx2, cy2 = _padded_bbox(
                x1, y1, x2, y2, img_w, img_h, self.crop_pad
            )

            # --- Crop output ---
            crop = img[cy1:cy2, cx1:cx2]
            crop_path = crops_dir / f"{stem}_{role}.jpg"
            cv2.imwrite(
                str(crop_path), crop,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            saved[f"crop_{role}"] = crop_path

            # --- Mask output (masked RGB object on black background, padded to square) ---
            mask_img = _crop_masked_object(img, det.mask, det.bbox)
            mask_path = masks_dir / f"{stem}_{role}.png"
            cv2.imwrite(str(mask_path), mask_img)
            saved[f"mask_{role}"] = mask_path

        return saved
