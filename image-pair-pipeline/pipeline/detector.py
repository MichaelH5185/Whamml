"""
detector.py
-----------
YOLO26 (ultralytics) segmentation inference wrapper.

Returns per-image Detection objects containing:
  - bbox:    [x1, y1, x2, y2] in pixel coordinates
  - mask:    Nx2 array of polygon points (pixel coordinates), or None
  - conf:    detection confidence (float)
  - cls_id:  YOLO class index (int)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

# Lazy import so the CLI can load without GPU / ultralytics installed
_model_cache: dict[str, object] = {}


@dataclass
class Detection:
    bbox: list[float]          # [x1, y1, x2, y2] in image pixel coordinates
    mask: Optional[np.ndarray] # shape (N, 2) polygon in image pixel coords or None
    conf: float
    cls_id: int
    img_size: tuple[int, int] = (1, 1)  # (img_w, img_h)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    def normalized_bbox(self) -> list[float]:
        """Return [x1, y1, x2, y2] normalized to [0, 1] relative to image width/height."""
        w, h = self.img_size
        if w <= 1 or h <= 1:
            return list(self.bbox)
        return [
            self.bbox[0] / float(w),
            self.bbox[1] / float(h),
            self.bbox[2] / float(w),
            self.bbox[3] / float(h),
        ]

    def normalized_area(self) -> float:
        """Area in normalized [0, 1] frame coordinates."""
        b = self.normalized_bbox()
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def load_model(weights: str | Path):
    """Load and cache a YOLO model from *weights*."""
    key = str(weights)
    if key not in _model_cache:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required: pip install ultralytics"
            ) from exc
        _model_cache[key] = YOLO(str(weights))
    return _model_cache[key]


def _poly_from_mask_tensor(mask_tensor, img_h: int, img_w: int) -> Optional[np.ndarray]:
    """
    Convert an ultralytics mask tensor (H×W binary) to a polygon (N×2 pixel coords).
    Returns None if the mask is empty.
    """
    # mask_tensor is a torch.Tensor or numpy array, values 0/1
    try:
        mask_np = mask_tensor.cpu().numpy() if hasattr(mask_tensor, "cpu") else np.array(mask_tensor)
    except Exception:
        return None

    mask_uint8 = (mask_np * 255).astype(np.uint8)
    # Resize to original image dimensions if needed
    if mask_uint8.shape != (img_h, img_w):
        mask_uint8 = cv2.resize(mask_uint8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Take largest contour
    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2).astype(np.float32)


def run_inference(
    image_path: str | Path,
    model,
    conf_threshold: float = 0.25,
    min_bbox_px: int = 50,
) -> list[Detection]:
    """
    Run YOLO segmentation inference on *image_path*.

    Parameters
    ----------
    image_path      Path to the image file.
    model           Loaded ultralytics YOLO model.
    conf_threshold  Minimum confidence score to keep.
    min_bbox_px     Minimum pixel dimension (width AND height) to keep.

    Returns
    -------
    List of Detection objects, filtered by size and confidence.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image to get dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_h, img_w = img.shape[:2]

    results = model(str(image_path), conf=conf_threshold, verbose=False)
    detections: list[Detection] = []

    for result in results:
        boxes = result.boxes
        masks = result.masks  # may be None for detection-only models

        if boxes is None:
            continue

        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1

            # Filter too-small detections
            if w < min_bbox_px or h < min_bbox_px:
                continue

            conf = float(box.conf[0].cpu())
            cls_id = int(box.cls[0].cpu())

            # Extract segmentation mask polygon if available
            poly: Optional[np.ndarray] = None
            if masks is not None and i < len(masks.data):
                poly = _poly_from_mask_tensor(masks.data[i], img_h, img_w)

            detections.append(Detection(
                bbox=[x1, y1, x2, y2],
                mask=poly,
                conf=conf,
                cls_id=cls_id,
                img_size=(img_w, img_h),
            ))

    return detections
