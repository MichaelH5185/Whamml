"""
matcher.py
----------
Pairs before/after images by their 4-digit UUID suffix, then finds the
best 1-to-1 assignment of detections across the pair using a size-normalised
IoU metric and the Hungarian algorithm.

Before filename pattern : frame_####.png  (or .jpg / .jpeg)
After  filename pattern : cesiumcamerasequence.####  (any extension)

The 4-digit suffix (####) is the shared key used to pair images.

Size-normalised IoU
-------------------
    intersection_area / min(area_A, area_B)

This is more lenient than standard IoU for objects that differ in size between
the before and after images (e.g. a building partially occluded in one shot),
while still requiring genuine spatial overlap.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection


# ---------------------------------------------------------------------------
# Filename utilities
# ---------------------------------------------------------------------------

# Before is Cesium reconstruction: cesiumcamerasequence.####
# After is original drone footage: frame_####
_BEFORE_PATTERN = re.compile(r"cesiumcamerasequence\.(\d{4})", re.IGNORECASE)
_AFTER_PATTERN  = re.compile(r"frame_(\d{4})", re.IGNORECASE)

# Flexible fallback patterns to handle either naming in either folder if needed:
_FLEXIBLE_PATTERN = re.compile(r"(?:frame_|cesiumcamerasequence\.)?(\d{4})", re.IGNORECASE)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _suffix(path: Path, primary_pattern: re.Pattern) -> Optional[str]:
    """Extract 4-digit suffix from filename using primary pattern or generic 4-digit fallback."""
    m = primary_pattern.search(path.name)
    if m:
        return m.group(1)
    # Fallback to any 4 digits in filename
    m2 = re.search(r"(\d{4})", path.name)
    return m2.group(1) if m2 else None


def scan_before_dir(before_dir: str | Path) -> dict[str, Path]:
    """Return {suffix: Path} for files in *before_dir* (cesiumcamerasequence.####)."""
    result: dict[str, Path] = {}
    for p in Path(before_dir).iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            s = _suffix(p, _BEFORE_PATTERN)
            if s:
                result[s] = p
    return result


def scan_after_dir(after_dir: str | Path) -> dict[str, Path]:
    """Return {suffix: Path} for files in *after_dir* (frame_####)."""
    result: dict[str, Path] = {}
    for p in Path(after_dir).iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            s = _suffix(p, _AFTER_PATTERN)
            if s:
                result[s] = p
    return result


def paired_suffixes(
    before_map: dict[str, Path],
    after_map: dict[str, Path],
    frame_step: int = 5,
    start_suffix: Optional[str] = None,
) -> list[str]:
    """
    Return sorted list of suffixes present in BOTH dirs, sampled every
    *frame_step* frames (by position in sorted order, not by numeric value).
    Extra after-frames that have no matching before-frame are silently ignored.

    Parameters
    ----------
    before_map   : Dict of {suffix: Path} for before images.
    after_map    : Dict of {suffix: Path} for after images.
    frame_step   : Stride for sampling frames.
    start_suffix : Optional 4-digit suffix to start processing from.
                   All frame pairs prior to this suffix are skipped.
    """
    common = sorted(set(before_map) & set(after_map))
    if start_suffix:
        # Standardize start_suffix (e.g. '36' -> '0036', 'frame_0036' -> '0036')
        s_norm = str(start_suffix).strip().lower()
        m = _BEFORE_PATTERN.search(s_norm) or _AFTER_PATTERN.search(s_norm) or re.search(r"(\d{1,4})", s_norm)
        if m:
            clean_suffix = m.group(1).zfill(4)
            common = [s for s in common if s >= clean_suffix]
        else:
            common = [s for s in common if s >= s_norm]

    # Sample every frame_step-th entry
    sampled = common[::frame_step]
    return sampled


# ---------------------------------------------------------------------------
# IoU calculations
# ---------------------------------------------------------------------------

def _intersection_area(a: list[float], b: list[float]) -> float:
    """Intersection area of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    w = max(0.0, ix2 - ix1)
    h = max(0.0, iy2 - iy1)
    return w * h


def calculate_iou(
    det_a: Detection,
    det_b: Detection,
    size_penalty_weight: float = 1.0,
) -> float:
    """
    Calculate IoU between two detections in normalized image coordinates [0, 1],
    properly penalizing size differences.

    Comparing boxes in normalized coordinates ensures that images with different
    resolutions (e.g. 4K drone before-images vs 1080p Cesium recreation after-images)
    are evaluated in the same frame of reference. Two boxes that do not occupy the
    same relative space in their respective images will have 0.0 intersection.
    """
    box_a = det_a.normalized_bbox()
    box_b = det_b.normalized_bbox()

    area_a = det_a.normalized_area()
    area_b = det_b.normalized_area()

    if area_a <= 0 or area_b <= 0:
        return 0.0

    inter = _intersection_area(box_a, box_b)
    if inter <= 0:
        return 0.0

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    standard_iou = inter / union

    if size_penalty_weight > 0.0:
        min_area = min(area_a, area_b)
        max_area = max(area_a, area_b)
        size_ratio = (min_area / max_area) if max_area > 0 else 0.0
        return float(standard_iou * (size_ratio ** size_penalty_weight))

    return float(standard_iou)


# Alias for backward compatibility
size_normalized_iou = calculate_iou


# ---------------------------------------------------------------------------
# Detection pair
# ---------------------------------------------------------------------------

class DetectionPair:
    """A matched before/after detection pair."""

    def __init__(
        self,
        suffix: str,
        before_path: Path,
        after_path: Path,
        det_before: Detection,
        det_after: Detection,
        iou: float,
    ):
        self.suffix = suffix
        self.before_path = before_path
        self.after_path = after_path
        self.det_before = det_before
        self.det_after = det_after
        self.iou = iou
        self.damage_class: Optional[int] = None   # filled by classifier / user
        self.building_lon: Optional[float] = None
        self.building_lat: Optional[float] = None
        self.building_address: Optional[str] = None
        self.match_distance_m: Optional[float] = None


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------

def match_detections(
    dets_before: list[Detection],
    dets_after: list[Detection],
    iou_threshold: float = 0.10,
    size_penalty_weight: float = 1.0,
) -> list[tuple[Detection, Detection, float]]:
    """
    Find the optimal 1-to-1 assignment between before and after detections
    using the Hungarian algorithm on size-penalized IoU scores.

    Returns list of (det_before, det_after, iou) for pairs above threshold.
    """
    if not dets_before or not dets_after:
        return []

    n_b = len(dets_before)
    n_a = len(dets_after)

    # Build cost matrix (we maximise IoU → minimise negative IoU)
    cost = np.zeros((n_b, n_a), dtype=float)
    for i, db in enumerate(dets_before):
        for j, da in enumerate(dets_after):
            cost[i, j] = -calculate_iou(db, da, size_penalty_weight=size_penalty_weight)

    row_ind, col_ind = linear_sum_assignment(cost)

    pairs = []
    for r, c in zip(row_ind, col_ind):
        iou = -cost[r, c]
        if iou >= iou_threshold:
            pairs.append((dets_before[r], dets_after[c], iou))

    return pairs
