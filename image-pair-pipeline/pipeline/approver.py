"""
approver.py
-----------
Interactive approval step for each frame pair.

For each sampled frame pair the pipeline:
  1. Runs YOLO on both images
  2. Picks the BEST matching detection pair (highest size-normalised IoU)
     — or the highest-confidence single detection if there is no overlap —
     as the EXAMPLE to show the user
  3. Renders a side-by-side visualisation:
       Left  : before image with best-match bbox + class label overlay
       Right : after  image with best-match bbox + class label overlay
  4. Saves the visualisation to  <output_crops>/approvals/<suffix>_preview.jpg
     and opens it with the OS default image viewer
  5. Prompts the user in the terminal:
       [y] keep all valid pairs for this frame pair
       [n] skip this frame pair
       [q] quit the pipeline immediately

Returns
-------
  "keep"  – process all pairs for this frame pair
  "skip"  – skip all pairs
  "quit"  – stop the entire pipeline
"""

from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detector import Detection
from .matcher import DetectionPair

# Distinct high-contrast color palette for matching pairs (BGR)
PAIR_PALETTE = [
    (255, 255, 0),    # Cyan / Light Blue
    (0, 255, 0),      # Bright Green
    (0, 140, 255),    # Orange
    (255, 0, 255),    # Magenta / Fuchsia
    (0, 230, 255),    # Yellow
    (30, 30, 255),    # Red
    (255, 170, 0),    # Deep Sky Blue
    (50, 205, 50),    # Lime Green
    (180, 105, 255),  # Hot Pink
    (0, 190, 255),    # Amber
    (255, 50, 150),   # Purple
    (200, 220, 0),    # Turquoise
]

CLASS_LABELS = {
    1: "1 – Negligible/Slight",
    2: "2 – Moderate",
    3: "3 – Substantial/Heavy",
    4: "4 – Very Heavy",
    5: "5 – Destruction",
    None: "? – No class match",
}


def _draw_pair_box(
    img: np.ndarray,
    bbox: list[float],
    mask: Optional[np.ndarray],
    pair_idx: int,
    color: tuple[int, int, int],
    damage_class: Optional[int],
    iou: Optional[float] = None,
    font_scale: float = 0.9,
    font_thick: int = 2,
    box_thick: int = 3,
) -> np.ndarray:
    """Draw bbox, prominent index badge, and optional mask contour directly on scaled canvas."""
    out = img.copy()
    img_h, img_w = out.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]

    # Draw bounding box
    cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)

    # Draw mask contour if available
    if mask is not None and len(mask) >= 3:
        pts = mask.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=box_thick)

    # Format label text
    cls_str = f"C{damage_class}" if damage_class is not None else "NoCls"
    if iou is not None:
        text = f" #{pair_idx}  {cls_str}  IoU={iou:.2f} "
    else:
        text = f" #{pair_idx}  {cls_str} "

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, font_scale, font_thick)

    pad_x = 6
    pad_y = 6
    badge_w = tw + pad_x * 2
    badge_h = th + base + pad_y * 2

    # Place badge above box if there is room; otherwise inside top of box
    if y1 - badge_h >= 0:
        bg_y1 = y1 - badge_h
        bg_y2 = y1
        text_y = y1 - base - pad_y
    else:
        bg_y1 = y1
        bg_y2 = min(img_h, y1 + badge_h)
        text_y = bg_y1 + th + pad_y

    bg_x1 = max(0, x1)
    bg_x2 = min(img_w, bg_x1 + badge_w)

    # Solid badge with dark outline for high contrast
    cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), 1)

    # Text color: black or white based on background brightness
    brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    txt_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)

    cv2.putText(
        out, text, (bg_x1 + pad_x, text_y),
        font, font_scale, txt_color, font_thick, cv2.LINE_AA,
    )
    return out


def _draw_unmatched_box(
    img: np.ndarray,
    bbox: list[float],
    conf: float,
    font_scale: float = 0.55,
    font_thick: int = 1,
) -> np.ndarray:
    """Draw a thin, muted grey box for an unmatched detection."""
    out = img.copy()
    img_h, img_w = out.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    color = (130, 130, 130)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)

    text = f" unmatched ({conf:.2f}) "
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, font_scale, font_thick)

    badge_h = th + base + 6
    bg_y1 = max(0, y1 - badge_h)
    bg_x1 = max(0, x1)
    bg_x2 = min(img_w, x1 + tw + 6)

    cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, y1), (50, 50, 50), -1)
    cv2.putText(
        out, text, (bg_x1 + 2, y1 - base - 2),
        font, font_scale, (210, 210, 210), font_thick, cv2.LINE_AA,
    )
    return out


def _prepare_canvas(img_path: Path, target_h: int) -> tuple[np.ndarray, float]:
    """Load image and scale it to target_h, returning (scaled_img, scale_factor)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((target_h, target_h, 3), dtype=np.uint8), 1.0
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_h, 3), dtype=np.uint8), 1.0
    scale = float(target_h) / float(h)
    target_w = max(1, int(round(w * scale)))
    scaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return scaled, scale


def render_approval_image(
    before_path: Path,
    after_path: Path,
    proposed_pairs: list[DetectionPair],
    dets_before: list[Detection],
    dets_after: list[Detection],
    suffix: str,
    target_h: int = 1080,
    building_points: Optional[list[dict]] = None,
) -> np.ndarray:
    """
    Build a side-by-side preview image showing all proposed matching pairs.

    Drawing is performed directly on the target-resolution canvas so text
    and index badges remain large, crisp, and fully readable without any
    downsampling blur.
    """
    # 1. Scale images to target_h FIRST
    left, scale_b = _prepare_canvas(before_path, target_h)
    right, scale_a = _prepare_canvas(after_path, target_h)

    # Dynamic font and stroke sizes based on canvas height
    font_scale = max(0.85, target_h / 1050.0)
    font_thick = max(2, int(round(font_scale * 2.2)))
    box_thick = max(3, int(round(font_scale * 3.0)))
    unmatched_font_scale = max(0.55, target_h / 1600.0)

    # Identify which detections are matched
    matched_before_bboxes = {tuple(p.det_before.bbox) for p in proposed_pairs}
    matched_after_bboxes  = {tuple(p.det_after.bbox) for p in proposed_pairs}

    # 2. Draw unmatched detections first (scaled to canvas)
    for det in dets_before:
        if tuple(det.bbox) not in matched_before_bboxes:
            s_bbox = [v * scale_b for v in det.bbox]
            left = _draw_unmatched_box(left, s_bbox, det.conf, font_scale=unmatched_font_scale)

    for det in dets_after:
        if tuple(det.bbox) not in matched_after_bboxes:
            s_bbox = [v * scale_a for v in det.bbox]
            right = _draw_unmatched_box(right, s_bbox, det.conf, font_scale=unmatched_font_scale)

    # 3. Draw proposed matching pairs with large, crisp colored badges
    for i, pair in enumerate(proposed_pairs, start=1):
        color = PAIR_PALETTE[(i - 1) % len(PAIR_PALETTE)]

        # Scale before detection to left canvas
        s_bbox_b = [v * scale_b for v in pair.det_before.bbox]
        s_mask_b = (pair.det_before.mask * scale_b) if pair.det_before.mask is not None else None
        left = _draw_pair_box(
            left, s_bbox_b, s_mask_b, pair_idx=i, color=color,
            damage_class=pair.damage_class, iou=pair.iou,
            font_scale=font_scale, font_thick=font_thick, box_thick=box_thick,
        )

        # Scale after detection to right canvas
        s_bbox_a = [v * scale_a for v in pair.det_after.bbox]
        s_mask_a = (pair.det_after.mask * scale_a) if pair.det_after.mask is not None else None
        right = _draw_pair_box(
            right, s_bbox_a, s_mask_a, pair_idx=i, color=color,
            damage_class=pair.damage_class, iou=pair.iou,
            font_scale=font_scale, font_thick=font_thick, box_thick=box_thick,
        )

    # 4. Draw building candidate points ON TOP of boxes/masks so they are clearly visible
    if building_points:
        for canvas, scale, tag in [(left, scale_b, "BEFORE"), (right, scale_a, "AFTER")]:
            overlay = canvas.copy()
            for idx, pt in enumerate(building_points):
                px = pt.get("pixel_x")
                py = pt.get("pixel_y")
                if px is None or py is None:
                    continue
                try:
                    sx = int(round(float(px) * scale))
                    sy = int(round(float(py) * scale))
                except (ValueError, TypeError):
                    continue

                if 0 <= sx < canvas.shape[1] and 0 <= sy < canvas.shape[0]:
                    # Large high-contrast bullseye marker
                    # Outer black ring
                    cv2.circle(canvas, (sx, sy), 14, (0, 0, 0), -1, cv2.LINE_AA)
                    # Bright neon yellow/cyan ring
                    cv2.circle(canvas, (sx, sy), 12, (0, 255, 255), -1, cv2.LINE_AA)
                    # Red inner bullseye
                    cv2.circle(canvas, (sx, sy), 6, (0, 0, 255), -1, cv2.LINE_AA)
                    # Crosshair ticks
                    cv2.line(canvas, (sx - 18, sy), (sx + 18, sy), (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.line(canvas, (sx, sy - 18), (sx, sy + 18), (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.line(canvas, (sx - 16, sy), (sx + 16, sy), (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(canvas, (sx, sy - 16), (sx, sy + 16), (0, 255, 255), 1, cv2.LINE_AA)

                    # Label with distance if available
                    dist = pt.get("distance_meters")
                    lbl = f"{dist:.0f}m" if dist is not None else f"P{idx+1}"
                    cv2.putText(
                        canvas, lbl, (sx + 14, sy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
                    )
                    cv2.putText(
                        canvas, lbl, (sx + 14, sy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
                    )

    # Vertical divider bar between before and after
    divider = np.full((target_h, 8, 3), (255, 255, 255), dtype=np.uint8)

    # Header banner with large, crisp text
    combined_w = left.shape[1] + 8 + right.shape[1]
    header_h = max(54, int(round(target_h * 0.055)))
    header = np.full((header_h, combined_w, 3), (25, 25, 25), dtype=np.uint8)
    n_pairs = len(proposed_pairs)
    banner_text = (
        f"Suffix {suffix}  |  {n_pairs} proposed pair(s) (matched by #index & color)  |  "
        f"BEFORE (left: {len(dets_before)} dets)  vs  AFTER (right: {len(dets_after)} dets)"
    )
    banner_scale = max(0.80, target_h / 1150.0)
    cv2.putText(
        header,
        banner_text,
        (16, int(round(header_h * 0.65))),
        cv2.FONT_HERSHEY_SIMPLEX, banner_scale, (240, 240, 240), 2, cv2.LINE_AA,
    )

    side_by_side = np.hstack([left, divider, right])
    preview = np.vstack([header, side_by_side])
    return preview


def open_image(path: Path) -> None:
    """Open *path* with the system default image viewer (non-blocking)."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        print(f"  [approver] Could not open image viewer: {exc}")


def parse_approval_input(
    user_input: str,
    max_index: int,
) -> tuple[str, dict[int, Optional[int]]]:
    """
    Parse approval input allowing both index selection and optional class assignment.

    Supports:
      "y" / "yes" / "all"         -> ("all", {})
      "y 3" / "all 3"             -> ("all", {1: 3, 2: 3, ...})
      "n" / "no" / "skip"         -> ("skip", {})
      "q" / "quit"                -> ("quit", {})
      "1"                         -> ("select", {1: None})
      "1, 3"                      -> ("select", {1: None, 3: None})
      "1:3"                       -> ("select", {1: 3})
      "1:3, 2:4"                  -> ("select", {1: 3, 2: 4})
      "1 c3"                      -> ("select", {1: 3})
      "1-3"                       -> ("select", {1: None, 2: None, 3: None})
      "1-3:2"                     -> ("select", {1: 2, 2: 2, 3: 2})
      "c3"                        -> ("all", {1: 3, ...})

    Returns:
      (action, overrides_dict) where overrides_dict maps 1-based index to class int (or None).
    """
    raw = user_input.strip()
    if not raw:
        return "invalid", {}

    lower = raw.lower()
    if lower in ("n", "no", "s", "skip"):
        return "skip", {}
    if lower in ("q", "quit", "exit"):
        return "quit", {}

    # Check for 'y', 'yes', 'all', optionally with a class e.g. 'y 3' or 'all c3'
    import re
    m_all = re.match(r"^(?:y|yes|all|a)(?:\s+c?([1-5]))?$", lower)
    if m_all:
        cls_override = int(m_all.group(1)) if m_all.group(1) else None
        overrides = {i: cls_override for i in range(1, max_index + 1)} if cls_override else {i: None for i in range(1, max_index + 1)}
        return "all", overrides

    # Check for standalone class shortcut e.g. 'c3'
    m_cls = re.match(r"^c([1-5])$", lower)
    if m_cls:
        cls_val = int(m_cls.group(1))
        return "all", {i: cls_val for i in range(1, max_index + 1)}

    # Parse index tokens e.g. '1:3, 2:4' or '1, 3' or '1 c3' or '1-2:3'
    cleaned = raw.replace("#", "")
    tokens = re.split(r"[,;]+|\s+(?=\d)", cleaned.strip())

    overrides: dict[int, Optional[int]] = {}
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        cls_val = None
        if ":" in token or "=" in token:
            parts = re.split(r"[:=]", token, maxsplit=1)
            idx_part = parts[0].strip()
            cls_str = parts[1].strip().lower().lstrip("c")
            if not cls_str.isdigit() or not (1 <= int(cls_str) <= 5):
                return "invalid", {}
            cls_val = int(cls_str)
        elif re.search(r"\s+c?[1-5]$", token, re.IGNORECASE):
            parts = re.split(r"\s+", token)
            idx_part = parts[0].strip()
            cls_str = parts[1].strip().lower().lstrip("c")
            if not cls_str.isdigit() or not (1 <= int(cls_str) <= 5):
                return "invalid", {}
            cls_val = int(cls_str)
        elif re.match(r"^\d+[cC][1-5]$", token):
            m = re.match(r"^(\d+)[cC]([1-5])$", token)
            idx_part = m.group(1)
            cls_val = int(m.group(2))
        else:
            idx_part = token

        # Parse idx_part (single index or range)
        if "-" in idx_part:
            r_parts = idx_part.split("-")
            if len(r_parts) != 2 or not r_parts[0].isdigit() or not r_parts[1].isdigit():
                return "invalid", {}
            start, end = int(r_parts[0]), int(r_parts[1])
            if start < 1 or end > max_index or start > end:
                return "invalid", {}
            for i in range(start, end + 1):
                overrides[i] = cls_val
        elif idx_part.isdigit():
            idx = int(idx_part)
            if idx < 1 or idx > max_index:
                return "invalid", {}
            overrides[idx] = cls_val
        else:
            return "invalid", {}

    if not overrides:
        return "invalid", {}

    return "select", overrides


def parse_index_selection(user_input: str, max_index: int) -> Optional[list[int]]:
    """Backward-compatible helper returning list of indices."""
    action, overrides = parse_approval_input(user_input, max_index)
    if action in ("all", "select"):
        return sorted(overrides.keys())
    return None


def prompt_approval(
    before_path: Path,
    after_path: Path,
    proposed_pairs: list[DetectionPair],
    dets_before: list[Detection],
    dets_after: list[Detection],
    suffix: str,
    approvals_dir: Path,
    preview_height: int = 1080,
    building_points: Optional[list[dict]] = None,
) -> tuple[str, list[DetectionPair]]:
    """
    Render + save + open the approval preview and block for user input.

    User can:
      - [y] keep all proposed pairs with existing classes
      - [y 3] keep all proposed pairs and assign them Class 3
      - [1:3, 2:4] keep specific pairs and assign specific classes
      - [1, 3] keep specific pairs with existing classes
      - [n] skip this frame pair
      - [q] quit the pipeline

    Returns
    -------
    (action, kept_pairs)
      action: "keep" | "skip" | "quit"
      kept_pairs: list of DetectionPair objects selected by user
    """
    approvals_dir.mkdir(parents=True, exist_ok=True)
    preview_path = approvals_dir / f"{suffix}_preview.jpg"

    preview = render_approval_image(
        before_path, after_path,
        proposed_pairs, dets_before, dets_after,
        suffix, preview_height,
        building_points=building_points,
    )
    cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"\n{'='*68}")
    print(f"  Frame suffix : {suffix}")
    print(f"  Before       : {before_path.name}")
    print(f"  After        : {after_path.name}")
    if building_points:
        print(f"  Building JSON: {len(building_points)} candidate point(s) plotted (crosshairs)")
    else:
        print(f"  Building JSON: [warn] No candidate points found for key '{suffix}' or '{before_path.stem.lower()}'")
    if proposed_pairs:
        print(f"  Proposed matches ({len(proposed_pairs)} pair{'s' if len(proposed_pairs) > 1 else ''}):")
        for i, p in enumerate(proposed_pairs, start=1):
            cls_lbl = CLASS_LABELS.get(p.damage_class, "? – No building match")
            conf_str = f"conf(B={p.det_before.conf:.2f}, A={p.det_after.conf:.2f})"
            addr_str = f" | {p.building_address}" if p.building_address else ""
            dist_str = f" (dist={p.match_distance_m:.1f}m)" if p.match_distance_m is not None else ""
            print(f"    [#{i}] IoU={p.iou:.2f} | Class={cls_lbl}{addr_str}{dist_str} | {conf_str}")
        print(f"  Options      : [y] keep all  |  [y <class>] keep all as class 1-5 (e.g. 'y 3')")
        print(f"                 [index] keep specific pairs (e.g. '1', '1, 3', or '1-2')")
        print(f"                 [index:class] assign class directly (e.g. '1:3' or '1:3, 2:4')")
        print(f"                 [n] skip frame  |  [q] quit pipeline")
    else:
        print(f"  No matching detection pairs found for this frame")
        print(f"  Options      : [n] skip frame  |  [q] quit pipeline")
    print(f"  Preview image: {preview_path}")
    print(f"{'='*68}")

    open_image(preview_path)

    while True:
        try:
            if proposed_pairs:
                prompt_msg = f"  [y/y <cls>/1:cls/n/q] ({len(proposed_pairs)} pair{'s' if len(proposed_pairs) > 1 else ''}) → "
            else:
                prompt_msg = "  [n] skip / [q] quit → "
            raw_choice = input(prompt_msg).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Interrupted — quitting.")
            return "quit", []

        action, overrides = parse_approval_input(raw_choice, len(proposed_pairs) if proposed_pairs else 1)

        if action == "quit":
            return "quit", []

        if action == "skip":
            print("  -> Skipped frame.")
            return "skip", []

        if action in ("all", "select"):
            kept: list[DetectionPair] = []
            log_parts = []
            for idx in sorted(overrides.keys()):
                pair = proposed_pairs[idx - 1]
                override_cls = overrides[idx]
                if override_cls is not None:
                    pair.damage_class = override_cls
                kept.append(pair)
                cls_display = f"C{pair.damage_class}" if pair.damage_class is not None else "NoCls"
                log_parts.append(f"#{idx}:{cls_display}")

            summary_str = ", ".join(log_parts)
            print(f"  -> Approved {len(kept)} pair(s): [{summary_str}].")
            return "keep", kept

        # Invalid entry
        if proposed_pairs:
            print(
                f"  Invalid selection. Please enter:\n"
                f"    - 'y' to keep all pairs with current classes\n"
                f"    - 'y 3' to keep all pairs and assign Class 3\n"
                f"    - '1, 3' to keep pairs 1 and 3\n"
                f"    - '1:3' or '1:3, 2:4' to select pairs and assign specific classes\n"
                f"    - 'n' to skip, or 'q' to quit."
            )
        else:
            print("  Please enter 'n' to skip or 'q' to quit.")
