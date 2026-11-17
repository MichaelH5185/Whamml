"""
cli.py
------
Main Click entrypoint for the before/after image pair pipeline.

Usage
-----
    python -m pipeline \\
        --weights      yolo26n-seg.pt \\
        --before-dir   ./before/ \\
        --after-dir    ./after/ \\
        --intrinsics-csv  camera_intrinsics.csv \\
        --gps-csv         camera_gps.csv \\
        --building-csv    buildings.csv \\
        --output-crops    ./output_crops/ \\
        --output-masks    ./output_masks/

The two per-frame CSVs are:
  intrinsics-csv : #name, x, y, alt, heading, pitch, roll, f, px, py, k1, k2, k3
  gps-csv        : #name, lon, lat, alt, omega, phi, kappa
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from tqdm import tqdm

from .detector import load_model, run_inference, Detection
from .geo import pixel_to_lonlat
from .classifier import BuildingIndex, load_frame_buildings_json, match_detection_to_point
from .matcher import (
    scan_before_dir, scan_after_dir, paired_suffixes,
    match_detections, DetectionPair,
)
from .dedup import deduplicate
from .exporter import Exporter
from .approver import prompt_approval


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str, comment: str = "#") -> pd.DataFrame:
    """Load a CSV, stripping the leading # from the first column header."""
    df = pd.read_csv(path, comment=None, sep=None, engine="python")
    # Strip '#' from first column name if present
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def load_intrinsics(csv_path: str) -> pd.DataFrame:
    """
    Load camera intrinsics/extrinsics CSV.
    Expected columns: name, heading, pitch, roll, f
    Returns DataFrame indexed by image basename (without extension).
    """
    df = _load_csv(csv_path)
    name_col = _find_col(df, ["name", "#name", "filename", "image"])
    df["_key"] = df[name_col].apply(lambda x: Path(str(x)).stem.lower())
    df = df.set_index("_key")
    return df


def load_gps(csv_path: str) -> pd.DataFrame:
    """
    Load GPS CSV.
    Expected columns: name, lon, lat, alt
    Returns DataFrame indexed by image basename stem.
    """
    df = _load_csv(csv_path)
    name_col = _find_col(df, ["name", "#name", "filename", "image"])
    df["_key"] = df[name_col].apply(lambda x: Path(str(x)).stem.lower())
    df = df.set_index("_key")
    return df


def load_class_map(path: Optional[str]) -> dict[str, int]:
    """
    Load a mapping file of frame suffix or name -> class number (1–5).
    Supports CSV (headers like frame/suffix and class/rating) or
    simple text files with lines like '0036: 3' or 'frame_0036, 3'.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Class map file not found: {path}")

    import re
    from .classifier import _parse_rating

    mapping: dict[str, int] = {}

    # Try CSV parser first
    try:
        df = _load_csv(str(p))
        name_col = None
        for cand in ["frame", "suffix", "name", "#name", "image", "id"]:
            for c in df.columns:
                if cand == c.lower():
                    name_col = c
                    break
            if name_col:
                break

        cls_col = None
        for cand in ["class", "rating", "damage", "damage_class"]:
            for c in df.columns:
                if cand == c.lower():
                    cls_col = c
                    break
            if cls_col:
                break

        if name_col and cls_col:
            for _, row in df.iterrows():
                val = _parse_rating(str(row[cls_col]))
                if val is not None:
                    raw_name = str(row[name_col]).strip()
                    m = re.search(r"(\d{4})", raw_name)
                    if m:
                        mapping[m.group(1)] = val
                    mapping[Path(raw_name).stem.lower()] = val
            if mapping:
                return mapping
    except Exception:
        pass

    # Fallback to line-by-line parsing
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[:,\t\s]+", line)
            if len(parts) >= 2:
                raw_name = parts[0].strip()
                val = _parse_rating(parts[1].strip())
                if val is not None:
                    m = re.search(r"(\d{4})", raw_name)
                    if m:
                        mapping[m.group(1)] = val
                    mapping[Path(raw_name).stem.lower()] = val

    return mapping


def _get_frame_params(
    stem: str,
    intrinsics_df: pd.DataFrame,
    gps_df: pd.DataFrame,
) -> Optional[dict]:
    """
    Look up camera parameters for a frame stem (e.g. 'frame_0001').
    Returns a dict with keys: lon, lat, alt, heading, pitch, roll, f
    or None if the frame is not found in either CSV.
    """
    key = stem.lower()
    if key not in gps_df.index or key not in intrinsics_df.index:
        return None

    gps_row = gps_df.loc[key]
    intr_row = intrinsics_df.loc[key]

    def _get(row, candidates, default=None):
        for cand in candidates:
            if cand.lower() in [c.lower() for c in row.index]:
                lmap = {c.lower(): c for c in row.index}
                return float(row[lmap[cand.lower()]])
        return default

    return {
        "lon":     _get(gps_row,  ["lon", "longitude"]),
        "lat":     _get(gps_row,  ["lat", "latitude"]),
        "alt":     _get(gps_row,  ["alt", "altitude"]),
        "heading": _get(intr_row, ["heading", "yaw"],   default=0.0),
        "pitch":   _get(intr_row, ["pitch"],             default=90.0),
        "roll":    _get(intr_row, ["roll"],              default=0.0),
        "f":       _get(intr_row, ["f", "focal", "focal_length"], default=22.2),
    }


def _get_ground_alt(gps_df: pd.DataFrame) -> float:
    """
    Return the ground altitude reference: altitude of the first frame
    in sorted order (assumed to be on the ground or near-ground).
    """
    alt_col_candidates = ["alt", "altitude"]
    alt_col = None
    for cand in alt_col_candidates:
        lmap = {c.lower(): c for c in gps_df.columns}
        if cand in lmap:
            alt_col = lmap[cand]
            break
    if alt_col is None:
        raise KeyError("Altitude column not found in GPS CSV")

    # Sort by index (frame name) to get first frame
    sorted_df = gps_df.sort_index()
    return float(sorted_df[alt_col].iloc[0])


# ---------------------------------------------------------------------------
# Detection → lon/lat projection
# ---------------------------------------------------------------------------

def detection_lonlat(
    det: Detection,
    img_w: int,
    img_h: int,
    params: dict,
    ground_alt: float,
    sensor_width_mm: float,
) -> Optional[tuple[float, float]]:
    """
    Project the centroid of *det* to a (lon, lat) coordinate.
    Returns None if projection fails.
    """
    cx, cy = det.center
    return pixel_to_lonlat(
        u=cx, v=cy,
        img_w=img_w, img_h=img_h,
        lon_cam=params["lon"],
        lat_cam=params["lat"],
        alt_cam=params["alt"],
        alt_ground=ground_alt,
        heading_deg=params["heading"],
        pitch_deg=params["pitch"],
        roll_deg=params["roll"],
        f_mm=params["f"],
        sensor_width_mm=sensor_width_mm,
    )


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

def _get_frame_candidates(
    frame_buildings_data: dict,
    suffix: str,
    before_stem: str,
    after_stem: str,
) -> "tuple[list, object]":
    """
    Try all known key variants for a frame's building candidates.

    The JSON keys are typically 'frame_####.jpg'. We expand many alias
    variants when loading, but also try them explicitly here for robustness.

    Returns (candidates_list, matched_key_string) or ([], None).
    """
    keys_to_try = [
        f"frame_{suffix}.jpg",
        f"frame_{suffix}.png",
        f"frame_{suffix}",
        suffix,
        after_stem,
        f"{after_stem}.jpg",
        f"{after_stem}.png",
        before_stem,
        f"{before_stem}.jpg",
    ]
    for k in keys_to_try:
        if not k:
            continue
        result = frame_buildings_data.get(k.lower())
        if result:
            return result, k
    return [], None


@click.command()
# ── Required inputs ──────────────────────────────────────────────────────────
@click.option("--weights",         required=True,  type=click.Path(exists=True), help="YOLO26 .pt weights file")
@click.option("--before-dir",      required=True,  type=click.Path(exists=True, file_okay=False), help="Directory of frame_#### images")
@click.option("--after-dir",       required=True,  type=click.Path(exists=True, file_okay=False), help="Directory of cesiumcamerasequence.#### images")
@click.option("--intrinsics-csv",  required=True,  type=click.Path(exists=True), help="Per-frame camera intrinsics CSV (heading, pitch, roll, f, …)")
@click.option("--gps-csv",         required=True,  type=click.Path(exists=True), help="Per-frame GPS CSV (lon, lat, alt, …)")
@click.option("--building-csv",    required=False, default=None, type=click.Path(exists=True), help="Per-building CSV (Lon, Lat, Rating, …)")
@click.option("--frame-buildings-json", default=None, type=click.Path(exists=True), help="JSON file with per-frame building pixel coordinates (pixel_x, pixel_y, damage_rating)")
# ── Output directories ────────────────────────────────────────────────────────
@click.option("--output-crops",  default="./output_crops/", show_default=True, type=click.Path(), help="Root output directory for padded crops")
@click.option("--output-masks",  default="./output_masks/", show_default=True, type=click.Path(), help="Root output directory for square segmentation masks")
# ── Sampling ─────────────────────────────────────────────────────────────────
@click.option("--frame-step",    default=5,    show_default=True, type=int,   help="Process every N-th frame pair (1=all, 5=every 5th)")
@click.option("--start-pair",    default=None, show_default=True, type=str,   help="Starting frame pair suffix or filename (e.g. '0036' or 'frame_0036')")
# ── Detection filtering ───────────────────────────────────────────────────────
@click.option("--conf",          default=0.25, show_default=True, type=float, help="YOLO confidence threshold")
@click.option("--min-bbox-px",   default=50,   show_default=True, type=int,   help="Minimum bbox side length in pixels")
# ── IoU matching ─────────────────────────────────────────────────────────────
@click.option("--iou-threshold", default=0.10, show_default=True, type=float, help="Minimum IoU score to keep a pair")
@click.option("--size-penalty-weight", default=1.0, show_default=True, type=float, help="Exponent for size similarity penalty: score = IoU * (min_area/max_area)^weight")
# ── Geo-projection ────────────────────────────────────────────────────────────
@click.option("--sensor-width-mm", default=13.2, show_default=True, type=float, help="Camera sensor width in mm (default: DJI 1\" sensor)")
@click.option("--building-radius-m", default=50.0, show_default=True, type=float, help="Maximum distance (m) to nearest building to assign a class")
@click.option("--distance-algorithm", default="raycast", show_default=True,
              type=click.Choice(["raycast", "fov", "hybrid", "drone"]),
              help="Distance/matching algorithm: 'raycast' (ground ray projection), 'fov' (line-of-sight directional matching), 'hybrid' (tries raycast then fov), or 'drone' (closest to drone GPS)")
# ── Crop/mask parameters ─────────────────────────────────────────────────────
@click.option("--crop-pad",      default=0.20, show_default=True, type=float, help="Fractional padding around bbox for crops (0.2 = 20%%)")
# ── Deduplication ────────────────────────────────────────────────────────────
@click.option("--max-per-building", default=3, show_default=True, type=int,   help="Maximum pairs kept per unique building")
# ── Class assignment overrides ────────────────────────────────────────────────
@click.option("--class-map",     default=None, type=click.Path(exists=True), help="CSV or text file mapping frame suffix/name to class (e.g. '0036,3')")
@click.option("--default-class", default=None, type=click.IntRange(1, 5),    help="Fallback class (1–5) when geolocation does not match a building")
@click.option("--force-class",   default=None, type=click.IntRange(1, 5),    help="Force all kept pairs to this class (1–5), bypassing building CSV")
# ── Approval preview ─────────────────────────────────────────────────────────
@click.option("--preview-height",   default=1080, show_default=True, type=int, help="Height of the approval preview image in pixels")
@click.option("--show-building-points", is_flag=True, default=False, help="Overlay candidate building points from JSON as crosshair markers on the preview")
@click.option("--no-approve",  is_flag=True, default=False, help="Skip interactive approval (process everything automatically)")
def main(
    weights, before_dir, after_dir, intrinsics_csv, gps_csv, building_csv,
    frame_buildings_json,
    output_crops, output_masks,
    frame_step, start_pair, conf, min_bbox_px, iou_threshold, size_penalty_weight,
    sensor_width_mm, building_radius_m, distance_algorithm,
    crop_pad, max_per_building,
    class_map, default_class, force_class,
    preview_height, show_building_points, no_approve,
):
    """
    Build Siamese-classifier training pairs from before/after drone imagery.

    For each sampled frame pair the pipeline:

    \b
    1. Runs YOLO26 segmentation on both images
    2. Matches detections using size-normalised IoU (Hungarian algorithm)
    3. Projects each detected building centroid to lon/lat
    4. Assigns a damage class from the building CSV (nearest neighbour)
    5. Shows an approval preview (y/n/q) unless --no-approve
    6. Deduplicates across frames (max N pairs per building)
    7. Exports crops + square masks to output_crops/ and output_masks/
    """
    click.echo("\n╔══════════════════════════════════════════════╗")
    click.echo("║  Before/After Siamese Pair Pipeline          ║")
    click.echo("╚══════════════════════════════════════════════╝\n")

    # ── Load model ──────────────────────────────────────────────────────────
    click.echo(f"[1/6] Loading YOLO model from {weights} …")
    model = load_model(weights)

    # ── Load CSVs ───────────────────────────────────────────────────────────
    click.echo(f"[2/6] Loading per-frame CSVs …")
    intrinsics_df = load_intrinsics(intrinsics_csv)
    gps_df        = load_gps(gps_csv)
    ground_alt    = _get_ground_alt(gps_df)
    click.echo(f"      Ground reference altitude: {ground_alt:.2f} m MSL")

    click.echo(f"[3/6] Loading building information …")
    frame_buildings_data = load_frame_buildings_json(frame_buildings_json)
    if frame_buildings_data:
        raw_keys = [k for k in frame_buildings_data if k.endswith(".jpg") or k.endswith(".png")]
        click.echo(f"      Loaded per-frame building locations from {frame_buildings_json}")
        click.echo(f"      → {len(raw_keys)} frame entries ({len(frame_buildings_data)} alias keys total)")
        sample = list(frame_buildings_data.keys())[:8]
        click.echo(f"      → Sample keys: {sample}")

    building_index = None
    if building_csv:
        click.echo(f"      Loading building CSV index (radius={building_radius_m} m) …")
        building_index = BuildingIndex(building_csv, match_radius_m=building_radius_m)
    elif not frame_buildings_data:
        click.echo("WARNING: Neither --building-csv nor --frame-buildings-json was provided. Damage classes will require manual override or --default-class.", err=True)

    # ── Load optional class overrides ───────────────────────────────────────
    class_map_dict = load_class_map(class_map)
    if class_map_dict:
        click.echo(f"      Loaded class map for {len(class_map_dict)} frame keys from {class_map}")
    if force_class is not None:
        click.echo(f"      Forcing all pairs to class {force_class}")
    elif default_class is not None:
        click.echo(f"      Default fallback class when unassigned: {default_class}")

    # ── Scan image directories ───────────────────────────────────────────────
    click.echo(f"[4/6] Scanning image directories …")
    before_map = scan_before_dir(before_dir)
    after_map  = scan_after_dir(after_dir)
    click.echo(f"      Before images found : {len(before_map)}")
    click.echo(f"      After  images found : {len(after_map)}")

    suffixes = paired_suffixes(before_map, after_map, frame_step=frame_step, start_suffix=start_pair)
    click.echo(f"      Paired suffixes     : {len(before_map) + len(after_map) - len(set(before_map) | set(after_map))} total common")
    if start_pair:
        click.echo(f"      Starting from pair  : {start_pair}")
    click.echo(f"      After step={frame_step} sampling : {len(suffixes)} pairs to inspect\n")

    if not suffixes:
        click.echo("ERROR: No matching frame pairs found.  Check --before-dir and --after-dir.", err=True)
        sys.exit(1)

    # ── Prepare output directories and exporter ─────────────────────────────
    approvals_dir = Path(output_crops) / "approvals"
    exporter = Exporter(output_crops, output_masks, crop_pad=crop_pad)

    # ── Main loop: per-frame processing ────────────────────────────────────
    click.echo("[5/6] Processing frame pairs …\n")

    all_approved_pairs: list[DetectionPair] = []
    stats = {"inspected": 0, "kept": 0, "skipped": 0, "dropped_no_match": 0}

    for suffix in tqdm(suffixes, desc="Frame pairs", unit="pair"):
        before_path = before_map[suffix]
        after_path  = after_map[suffix]

        # Look up camera params
        # Note: per-frame CSVs and frame_buildings.json use frame_#### (or suffix ####)
        # Check both before_path, after_path, and suffix for matching metadata
        before_stem = before_path.stem.lower()
        after_stem  = after_path.stem.lower()
        params = (
            _get_frame_params(before_stem, intrinsics_df, gps_df)
            or _get_frame_params(after_stem, intrinsics_df, gps_df)
            or _get_frame_params(f"frame_{suffix}", intrinsics_df, gps_df)
        )
        if params is None:
            tqdm.write(f"  [warn] No camera params for {before_stem} / {after_stem}, skipping.")
            continue

        # Run YOLO on both images
        try:
            dets_before = run_inference(before_path, model, conf, min_bbox_px)
            dets_after  = run_inference(after_path,  model, conf, min_bbox_px)
        except Exception as exc:
            tqdm.write(f"  [error] Inference failed for suffix {suffix}: {exc}")
            continue

        # Match detections
        raw_pairs = match_detections(
            dets_before, dets_after,
            iou_threshold=iou_threshold,
            size_penalty_weight=size_penalty_weight,
        )

        # Build DetectionPair objects + assign classes
        candidate_pairs: list[DetectionPair] = []
        import cv2 as _cv2
        _img_b = _cv2.imread(str(before_path))
        img_h, img_w = (_img_b.shape[:2] if _img_b is not None else (1080, 1920))

        for det_b, det_a, iou in raw_pairs:
            dp = DetectionPair(suffix, before_path, after_path, det_b, det_a, iou)

            # Determine class in order of precedence:
            # 1. Force class flag
            assigned_class = force_class

            # 2. Frame-level class map (by 4-digit suffix or stem)
            if assigned_class is None:
                assigned_class = (
                    class_map_dict.get(suffix)
                    or class_map_dict.get(before_stem)
                    or class_map_dict.get(after_stem)
                    or class_map_dict.get(f"frame_{suffix}")
                )

            # 3. Exact Point-in-Mask Lookup from Per-Frame Building JSON
            # Check candidate points for frame_#### or suffix ####
            bldg_info = None
            if assigned_class is None and frame_buildings_data:
                frame_candidates, _matched_key = _get_frame_candidates(
                    frame_buildings_data, suffix, before_stem, after_stem
                )
                if not frame_candidates:
                    tqdm.write(f"  [json] No candidates for suffix {suffix!r} "
                               f"(tried: frame_{suffix}.jpg, frame_{suffix}, {after_stem}, {suffix})")
                else:
                    tqdm.write(f"  [json] Found {len(frame_candidates)} candidate(s) via key {_matched_key!r}")
                # Test matching on both det_b and det_a (whichever is the frame where pixel coordinates apply)
                matched_cand = match_detection_to_point(det_b, frame_candidates)
                if matched_cand is None:
                    matched_cand = match_detection_to_point(det_a, frame_candidates)

                if matched_cand is not None:
                    bldg_info = matched_cand
                    assigned_class = matched_cand.get("class")
                    dp.building_address = matched_cand.get("address", "")
                    if matched_cand.get("distance_meters") is not None:
                        dp.match_distance_m = float(matched_cand["distance_meters"])
                    if matched_cand.get("handle"):
                        dp.building_address = f"{dp.building_address} (ID: {matched_cand['handle']})".strip()

            # 4. Geolocation and Building Distance Lookup (only if building_index was provided)
            d_lon, d_lat = params.get("lon"), params.get("lat")
            heading = params.get("heading", 0.0)
            u_norm = (det_b.center[0] / img_w) if img_w > 0 else 0.5

            if assigned_class is None and building_index is not None and distance_algorithm in ("raycast", "hybrid"):
                lonlat = detection_lonlat(det_b, img_w, img_h, params, ground_alt, sensor_width_mm)
                if lonlat is not None:
                    lon, lat = lonlat
                    dp.building_lon = lon
                    dp.building_lat = lat
                    bldg_info = building_index.nearest_building(lon, lat)
                    dp.building_address = bldg_info["address"]
                    dp.match_distance_m = bldg_info["distance_m"]
                    if assigned_class is None and bldg_info["distance_m"] <= building_radius_m:
                        assigned_class = bldg_info["class"]

            if assigned_class is None and building_index is not None and (distance_algorithm == "fov" or (distance_algorithm == "hybrid" and assigned_class is None)) and d_lon is not None and d_lat is not None:
                # FOV directional matching: find building along camera sightline
                fov_info = building_index.nearest_in_fov(
                    drone_lon=d_lon,
                    drone_lat=d_lat,
                    heading_deg=heading,
                    u_norm=u_norm,
                    hfov_deg=65.0,
                    max_dist_m=max(1500.0, building_radius_m * 2.0),
                )
                if fov_info is not None:
                    bldg_info = fov_info
                    dp.building_address = f"{fov_info['address']} (FOV bearing {fov_info['bearing_deg']:.0f}°)"
                    dp.match_distance_m = fov_info["distance_m"]
                    dp.building_lon = fov_info["building_lon"]
                    dp.building_lat = fov_info["building_lat"]
                    if assigned_class is None:
                        assigned_class = fov_info["class"]

            if assigned_class is None and building_index is not None and (distance_algorithm == "drone" or (distance_algorithm in ("raycast", "hybrid") and assigned_class is None and (bldg_info is None or bldg_info["distance_m"] > building_radius_m))):
                if d_lon is not None and d_lat is not None:
                    drone_bldg = building_index.nearest_building(d_lon, d_lat)
                    if drone_bldg["distance_m"] <= building_radius_m or distance_algorithm == "drone":
                        assigned_class = drone_bldg["class"]
                        dp.building_address = f"{drone_bldg['address']} (drone GPS)"
                        dp.match_distance_m = drone_bldg["distance_m"]
                        if dp.building_lon is None:
                            dp.building_lon = drone_bldg["building_lon"]
                            dp.building_lat = drone_bldg["building_lat"]

            # 5. Fallback: default-class flag
            if assigned_class is None and default_class is not None:
                assigned_class = default_class

            dp.damage_class = assigned_class
            candidate_pairs.append(dp)

        stats["inspected"] += 1

        # ── Approval step ──────────────────────────────────────────────────
        frame_pts = None
        if frame_buildings_data:
            frame_pts, _fp_key = _get_frame_candidates(
                frame_buildings_data, suffix, before_stem, after_stem
            )
            frame_pts = frame_pts or None

        if not no_approve:
            decision, approved_for_frame = prompt_approval(
                before_path=before_path,
                after_path=after_path,
                proposed_pairs=candidate_pairs,
                dets_before=dets_before,
                dets_after=dets_after,
                suffix=suffix,
                approvals_dir=approvals_dir,
                preview_height=preview_height,
                building_points=frame_pts if show_building_points else None,
            )

            if decision == "quit":
                click.echo("\nQuitting at user request.")
                break
            if decision == "skip":
                stats["skipped"] += 1
                continue
        else:
            approved_for_frame = candidate_pairs

        # ── Add approved pairs ─────────────────────────────────────────────
        valid_kept = [p for p in approved_for_frame if p.damage_class is not None]
        all_approved_pairs.extend(valid_kept)
        stats["kept"] += len(valid_kept)

    # ── Deduplication ───────────────────────────────────────────────────────
    click.echo(f"\n[6/6] Deduplicating {len(all_approved_pairs)} approved pairs "
               f"(max {max_per_building} per building) …")
    final_pairs = deduplicate(all_approved_pairs, max_per_building=max_per_building)
    click.echo(f"      After dedup: {len(final_pairs)} pairs remain.\n")

    # ── Export ──────────────────────────────────────────────────────────────
    click.echo("      Exporting crops and masks …")
    export_stats: dict[int, int] = {}

    for pair in tqdm(final_pairs, desc="Exporting", unit="pair"):
        bkey = (round(pair.building_lon or 0, 7), round(pair.building_lat or 0, 7))
        try:
            exporter.export_pair(
                suffix=pair.suffix,
                damage_class=pair.damage_class,
                before_path=pair.before_path,
                after_path=pair.after_path,
                det_before=pair.det_before,
                det_after=pair.det_after,
                building_key=bkey,
            )
            export_stats[pair.damage_class] = export_stats.get(pair.damage_class, 0) + 1
        except Exception as exc:
            tqdm.write(f"  [error] Export failed for suffix {pair.suffix}: {exc}")

    # ── Summary ─────────────────────────────────────────────────────────────
    click.echo("\n╔══════════════════════════════════════════════╗")
    click.echo("║  Pipeline complete — summary                  ║")
    click.echo("╠══════════════════════════════════════════════╣")
    click.echo(f"║  Frame pairs inspected  : {stats['inspected']:>6}              ║")
    click.echo(f"║  Pairs approved (raw)   : {stats['kept']:>6}              ║")
    click.echo(f"║  Pairs skipped          : {stats['skipped']:>6}              ║")
    click.echo(f"║  Dropped (no building)  : {stats['dropped_no_match']:>6}              ║")
    click.echo(f"║  Final pairs (post-dedup): {len(final_pairs):>5}              ║")
    click.echo("╠══════════════════════════════════════════════╣")
    click.echo("║  Class distribution:                          ║")
    for cls in sorted(export_stats):
        click.echo(f"║    class{cls} : {export_stats[cls]:>5} pairs                    ║")
    click.echo("╠══════════════════════════════════════════════╣")
    click.echo(f"║  Crops  → {Path(output_crops).resolve()}")
    click.echo(f"║  Masks  → {Path(output_masks).resolve()}")
    click.echo("╚══════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
