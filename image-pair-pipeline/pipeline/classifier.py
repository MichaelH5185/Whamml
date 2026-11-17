"""
classifier.py
-------------
Assigns a damage class (1–5) to a detection by finding the nearest entry
in the building CSV using a KD-tree on projected ENU coordinates.

Building CSV expected columns (case-insensitive):
    Lon / lon / longitude
    Lat / lat / latitude
    Rating / rating / class / Class

Rating text → integer class mapping
    1  Negligible to Slight / Slight / Negligible / None
    2  Moderate
    3  Substantial / Substantial to Heavy
    4  Very Heavy / Heavy
    5  Destruction / Complete / Total Loss
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .geo import latlon_to_enu_array, haversine_m
import cv2


# ---------------------------------------------------------------------------
# Frame building JSON lookup
# ---------------------------------------------------------------------------

def load_frame_buildings_json(json_path: Optional[str]) -> dict[str, list[dict]]:
    """
    Load a JSON file containing per-frame building pixel coordinates and ratings.
    Keys in the JSON may be exact filenames (e.g. 'frame_0001.jpg', 'frame_0001.png'),
    stems ('frame_0001'), or 4-digit suffixes ('0001').

    Standardizes all variants into the dictionary so any lookup variant works.
    """
    if not json_path:
        return {}
    import json
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Frame buildings JSON not found: {json_path}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If the JSON is wrapped (e.g. {"metadata": ..., "frames": {...}} or {"images": {...}})
    if isinstance(data, dict):
        for candidate_key in ("frames", "images", "data"):
            if candidate_key in data and isinstance(data[candidate_key], dict):
                data = data[candidate_key]
                break

    result: dict[str, list[dict]] = {}
    import re

    # If data is a list of frame items e.g. [{"frame": "frame_0001.jpg", "buildings": [...]}, ...]
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            frame_id = item.get("frame") or item.get("image") or item.get("name") or item.get("id")
            bldgs = item.get("buildings") or item.get("annotations") or item.get("objects") or [item]
            if frame_id and isinstance(bldgs, list):
                _index_frame_entry(result, str(frame_id), bldgs)
        return result

    if not isinstance(data, dict):
        return result

    for key, bldgs in data.items():
        if not isinstance(bldgs, list):
            continue
        _index_frame_entry(result, str(key), bldgs)

    return result


def _index_frame_entry(result: dict[str, list[dict]], key_str: str, bldgs: list[dict]) -> None:
    import re
    k_str = key_str.strip().lower()
    # 1. Exact raw key as in JSON
    result[k_str] = bldgs
    # 2. Key without extension if it has one (e.g. 'frame_0001.jpg' -> 'frame_0001')
    k_stem = Path(k_str).stem.lower()
    result[k_stem] = bldgs
    # 3. Key with .jpg added if not present (e.g. 'frame_0001' -> 'frame_0001.jpg')
    if not k_str.endswith(".jpg"):
        result[f"{k_stem}.jpg"] = bldgs
    # 4. Key with .png added
    if not k_str.endswith(".png"):
        result[f"{k_stem}.png"] = bldgs

    # 5. Extract 4-digit suffix and map '####', 'frame_####', 'frame_####.jpg', 'frame_####.png'
    m = re.search(r"(\d{4})", k_str)
    if m:
        s = m.group(1)
        result[s] = bldgs
        result[f"frame_{s}"] = bldgs
        result[f"frame_{s}.jpg"] = bldgs
        result[f"frame_{s}.png"] = bldgs


def match_detection_to_point(
    det,
    candidates: list[dict],
) -> Optional[dict]:
    """
    Find which building candidate point (pixel_x, pixel_y) fits within the bounds of the detection mask
    (or bounding box if mask is absent).

    If multiple points fit within the mask/bounds, preference is given to the closest point using the
    'distance_meters' field of the candidate object (with ties broken by distance to detection center).

    Returns the best matched building candidate dict with parsed 'class', or None.
    """
    if not candidates:
        return None

    # Get detection center and bounds
    cx, cy = det.center
    inside_matches: list[tuple[float, float, dict]] = []

    # Check mask polygon if available
    poly = None
    if det.mask is not None and len(det.mask) >= 3:
        poly = det.mask.astype(np.float32).reshape((-1, 1, 2))

    bx1, by1, bx2, by2 = det.bbox

    for cand in candidates:
        px = cand.get("pixel_x")
        py = cand.get("pixel_y")
        if px is None or py is None:
            continue
        try:
            px = float(px)
            py = float(py)
        except (ValueError, TypeError):
            continue

        # Fast reject if outside bounding box
        if not (bx1 <= px <= bx2 and by1 <= py <= by2):
            continue

        # Point in mask polygon test
        is_inside = False
        if poly is not None:
            # cv2.pointPolygonTest returns >= 0 if inside or on contour
            dist = cv2.pointPolygonTest(poly, (float(px), float(py)), measureDist=False)
            is_inside = (dist >= 0)
        else:
            # Fallback to bbox bounds
            is_inside = True

        if is_inside:
            # Distance from drone/camera (distance_meters)
            dist_m = cand.get("distance_meters")
            try:
                dist_m_val = float(dist_m) if dist_m is not None else float("inf")
            except (ValueError, TypeError):
                dist_m_val = float("inf")

            # Secondary tie-breaker: pixel distance to detection center
            dist_to_center = (px - cx) ** 2 + (py - cy) ** 2

            inside_matches.append((dist_m_val, dist_to_center, cand))

    if not inside_matches:
        return None

    # Sort primarily by closest distance_meters ascending, secondarily by center proximity
    inside_matches.sort(key=lambda x: (x[0], x[1]))
    best_cand = inside_matches[0][2].copy()

    # Parse rating -> class
    raw_rating = best_cand.get("damage_rating") or best_cand.get("rating") or best_cand.get("Rating") or ""
    best_cand["class"] = _parse_rating(str(raw_rating))
    best_cand["raw_rating"] = str(raw_rating)
    best_cand["address"] = best_cand.get("address", "")
    best_cand["handle"] = best_cand.get("handle", "")
    return best_cand


# ---------------------------------------------------------------------------
# Rating text → class integer
# ---------------------------------------------------------------------------

RATING_MAP: dict[str, int] = {
    "none": 1,
    "negligible": 1,
    "slight": 1,
    "negligible to slight": 1,
    "minor": 1,
    "moderate": 2,
    "substantial": 3,
    "substantial to heavy": 3,
    "heavy": 3,
    "very heavy": 4,
    "destruction": 5,
    "complete": 5,
    "total loss": 5,
    "total": 5,
}


def _parse_rating(raw: str) -> Optional[int]:
    """Convert a rating string to a 1–5 integer class, or None if unknown."""
    key = str(raw).strip().lower()
    if not key:
        return None
    # Exact match first
    if key in RATING_MAP:
        return RATING_MAP[key]
    # Try numeric
    try:
        v = int(float(key))
        if 1 <= v <= 5:
            return v
    except ValueError:
        pass
    # Partial / prefix match
    for k, v in RATING_MAP.items():
        if key.startswith(k) or (len(key) >= 4 and k.startswith(key)):
            return v
    return None


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first column name in *candidates* that exists in *df* (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"Could not find any of {candidates} in columns: {list(df.columns)}")


# ---------------------------------------------------------------------------
# BuildingIndex
# ---------------------------------------------------------------------------

class BuildingIndex:
    """
    Spatial index over the building CSV for fast nearest-neighbour queries.

    Attributes
    ----------
    lons, lats      : Arrays of building coordinates.
    classes         : Integer class array (1–5), NaN-masked invalid rows.
    ref_lon/lat     : Reference point used for the ENU projection.
    tree            : scipy KDTree in ENU space (metres).
    """

    def __init__(self, building_csv: str, match_radius_m: float = 30.0):
        self.match_radius_m = match_radius_m
        df = pd.read_csv(building_csv, comment="#")

        lon_col = _find_col(df, ["Lon", "lon", "longitude", "long"])
        lat_col = _find_col(df, ["Lat", "lat", "latitude"])
        rating_col = _find_col(df, ["Rating", "rating", "damage", "class", "Class"])

        self.lons = df[lon_col].to_numpy(dtype=float)
        self.lats = df[lat_col].to_numpy(dtype=float)

        # Address / site name if available
        try:
            addr_col = _find_col(df, ["SITEADDR", "siteaddr", "address", "addr", "street", "location"])
            self.addresses = df[addr_col].astype(str).tolist()
        except KeyError:
            self.addresses = [""] * len(df)

        # Parse ratings → classes
        self.raw_ratings = df[rating_col].astype(str).tolist()
        self.classes = np.array(
            [_parse_rating(r) if _parse_rating(r) is not None else -1
             for r in self.raw_ratings],
            dtype=int,
        )

        # Reference point for ENU projection: centroid of all buildings
        self.ref_lon = float(np.mean(self.lons))
        self.ref_lat = float(np.mean(self.lats))

        enu = latlon_to_enu_array(self.lons, self.lats, self.ref_lon, self.ref_lat)
        self.tree = KDTree(enu)

        valid = np.sum(self.classes > 0)
        total = len(self.classes)
        print(f"[classifier] Loaded {total} buildings ({valid} with valid ratings).")

    def nearest_building(self, lon: float, lat: float) -> dict:
        """
        Query nearest building regardless of distance threshold.
        Returns dict with: class, distance_m, address, raw_rating, lon, lat.
        """
        from .geo import latlon_to_enu_array
        import numpy as _np

        q_enu = latlon_to_enu_array(
            _np.array([lon]), _np.array([lat]),
            self.ref_lon, self.ref_lat,
        )
        dist_enu, idx = self.tree.query(q_enu[0], k=1)
        true_dist = haversine_m(lon, lat, self.lons[idx], self.lats[idx])
        cls = self.classes[idx]

        return {
            "class": int(cls) if cls > 0 else None,
            "distance_m": float(true_dist),
            "address": self.addresses[idx],
            "raw_rating": self.raw_ratings[idx],
            "building_lon": float(self.lons[idx]),
            "building_lat": float(self.lats[idx]),
        }

    def nearest_in_fov(
        self,
        drone_lon: float,
        drone_lat: float,
        heading_deg: float,
        u_norm: float = 0.5,
        hfov_deg: float = 60.0,
        max_dist_m: float = 1200.0,
    ) -> Optional[dict]:
        """
        Field-of-View (FOV) Directional Matching Algorithm:
        Instead of projecting rays to an assumed flat elevation, this matches
        buildings in front of the drone along the camera's line-of-sight.

        Parameters
        ----------
        drone_lon, drone_lat : Drone coordinates.
        heading_deg          : Camera compass heading (0=North, 90=East).
        u_norm               : Normalized horizontal pixel coordinate [0=left, 1=right].
                               0.5 = center of the frame.
        hfov_deg             : Horizontal field of view in degrees (~60-70 deg for 22mm lens).
        max_dist_m           : Maximum look distance (meters).
        """
        from .geo import bearing_degrees, angular_difference_deg

        # Target azimuth for this detection column:
        # e.g. center pixel u_norm=0.5 -> offset = 0. Left pixel u_norm=0.0 -> offset = -hfov/2
        azimuth_offset = (u_norm - 0.5) * hfov_deg
        target_bearing = (heading_deg + azimuth_offset + 360.0) % 360.0

        best_score = float("inf")
        best_info = None

        # Filter candidate buildings within max_dist_m
        for i in range(len(self.lons)):
            blon, blat = self.lons[i], self.lats[i]
            d = haversine_m(drone_lon, drone_lat, blon, blat)
            if d > max_dist_m:
                continue

            brg = bearing_degrees(drone_lon, drone_lat, blon, blat)
            ang_diff = angular_difference_deg(brg, target_bearing)

            # Building must be within the camera view sector (+- half FOV plus small margin)
            if ang_diff > (hfov_deg / 2.0 + 15.0):
                continue

            # Composite cost: angular deviation (degrees) * weight + distance (meters) * weight
            # Buildings closer to the target ray line get much lower cost
            # Cross-track distance in meters: d * sin(ang_diff)
            cross_track_m = d * np.sin(np.radians(ang_diff))
            cost = cross_track_m * 2.0 + d * 0.1

            if cost < best_score:
                best_score = cost
                cls = self.classes[i]
                best_info = {
                    "class": int(cls) if cls > 0 else None,
                    "distance_m": float(d),
                    "bearing_deg": float(brg),
                    "angular_diff_deg": float(ang_diff),
                    "address": self.addresses[i],
                    "raw_rating": self.raw_ratings[i],
                    "building_lon": float(blon),
                    "building_lat": float(blat),
                }

        return best_info

    def query_details(self, lon: float, lat: float) -> Optional[dict]:
        """
        Return details of nearest building if within match_radius_m, else None.
        """
        info = self.nearest_building(lon, lat)
        if info["distance_m"] > self.match_radius_m:
            return None
        return info

    def query(self, lon: float, lat: float) -> Optional[int]:
        """
        Return the damage class (1–5) for the nearest building within
        *match_radius_m*, or None if no building is close enough.
        """
        details = self.query_details(lon, lat)
        return details["class"] if details is not None else None
