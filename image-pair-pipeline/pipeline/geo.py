"""
geo.py
------
Geographic projection utilities.

Converts a pixel coordinate (u, v) inside a drone image to an estimated
real-world (lon, lat) using:
  - Drone GPS position   : lon, lat, alt  (from gps_csv)
  - Camera orientation   : heading, pitch, roll  (from intrinsics_csv)
  - Camera intrinsics    : f  (focal length in mm, from intrinsics_csv)
  - Sensor width         : sensor_width_mm (CLI flag, default 13.2 mm for DJI 1" sensor)
  - Ground reference alt : altitude of first frame (assumed on ground)

Coordinate conventions
  Camera frame : X = right, Y = down, Z = forward (into scene)
  World ENU    : X = East,  Y = North, Z = Up
"""

from __future__ import annotations
from math import cos, sin, radians, degrees, sqrt, atan2, asin
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Camera rotation
# ---------------------------------------------------------------------------

def _rotation_matrix(heading_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build a 3×3 rotation matrix from camera frame (X=right, Y=down, Z=fwd)
    to world ENU frame (X=East, Y=North, Z=Up).

    Parameters
    ----------
    heading_deg : Compass heading 0–360 (0=North, 90=East).
    pitch_deg   : Camera tilt from horizontal; 90 = straight-down nadir.
    roll_deg    : Camera roll; 0 = level, positive = right side down.

    Notes
    -----
    For a nadir camera (pitch=90, roll=0) pointing North (heading=0):
        Camera X (right) → ENU East  [1,0,0]
        Camera Y (down)  → ENU South [0,-1,0]  (top of image = forward = North)
        Camera Z (fwd)   → ENU Down  [0,0,-1]
    """
    h = radians(heading_deg)
    p = radians(pitch_deg)
    r = radians(roll_deg)

    # Look direction (camera Z axis) in ENU
    # pitch=90 → look straight down; pitch=0 → look horizontally along heading
    look = np.array([
        cos(p) * sin(h),    # East
        cos(p) * cos(h),    # North
        -sin(p),            # Up  (negative → downward)
    ])

    # Right direction (camera X axis) in ENU — perpendicular to look, nearly horizontal
    # For heading h, right is 90° clockwise from forward in the horizontal plane
    raw_right = np.array([cos(h), -sin(h), 0.0])

    # Apply roll: rotate right around look axis
    if abs(roll_deg) > 1e-3:
        # Rodrigues rotation of raw_right around look by roll_deg
        k = look / (np.linalg.norm(look) + 1e-12)
        right = (
            raw_right * cos(r)
            + np.cross(k, raw_right) * sin(r)
            + k * np.dot(k, raw_right) * (1 - cos(r))
        )
    else:
        right = raw_right

    # Down direction (camera Y axis) = Z × X  in camera frame → cross in world
    down = np.cross(look, right)
    down = down / (np.linalg.norm(down) + 1e-12)

    # Columns: ENU coordinates of camera X, Y, Z axes
    R = np.column_stack([right, down, look])
    return R


# ---------------------------------------------------------------------------
# Haversine distance (meters)
# ---------------------------------------------------------------------------

def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return the great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000.0  # Earth radius in metres
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return 2 * R * asin(sqrt(a))


# ---------------------------------------------------------------------------
# ENU offset → lon/lat
# ---------------------------------------------------------------------------

def enu_offset_to_latlon(
    lon_origin: float, lat_origin: float,
    east_m: float, north_m: float,
) -> tuple[float, float]:
    """
    Convert (east_m, north_m) ENU offset from (lon_origin, lat_origin) to
    a new (lon, lat).
    """
    R = 6_371_000.0
    lat_rad = radians(lat_origin)
    dlat = north_m / R
    dlon = east_m / (R * cos(lat_rad) + 1e-12)
    return lon_origin + degrees(dlon), lat_origin + degrees(dlat)


enu_offset_to_lonlat = enu_offset_to_latlon


def bearing_degrees(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate compass bearing in degrees (0–360, 0=North, 90=East) from
    (lon1, lat1) to (lon2, lat2).
    """
    dlon = radians(lon2 - lon1)
    lat1_r, lat2_r = radians(lat1), radians(lat2)
    y = sin(dlon) * cos(lat2_r)
    x = cos(lat1_r) * sin(lat2_r) - sin(lat1_r) * cos(lat2_r) * cos(dlon)
    return (degrees(atan2(y, x)) + 360.0) % 360.0


def angular_difference_deg(angle1: float, angle2: float) -> float:
    """Return shortest angular difference in degrees between two compass bearings [0, 180]."""
    diff = (angle1 - angle2 + 180.0) % 360.0 - 180.0
    return abs(diff)


# ---------------------------------------------------------------------------
# Main projection function
# ---------------------------------------------------------------------------

def pixel_to_lonlat(
    u: float,
    v: float,
    img_w: int,
    img_h: int,
    lon_cam: float,
    lat_cam: float,
    alt_cam: float,
    alt_ground: float,
    heading_deg: float,
    pitch_deg: float,
    roll_deg: float,
    f_mm: float,
    sensor_width_mm: float = 13.2,
) -> Optional[tuple[float, float]]:
    """
    Project pixel (u, v) to estimated ground-level (lon, lat).

    Parameters
    ----------
    u, v             : Pixel coordinates (u=col from left, v=row from top).
    img_w, img_h     : Image dimensions in pixels.
    lon_cam, lat_cam : Drone GPS coordinates.
    alt_cam          : Drone GPS altitude (metres MSL).
    alt_ground       : Ground elevation (metres MSL); use first-frame altitude.
    heading_deg      : Camera heading (compass degrees, 0=North).
    pitch_deg        : Camera tilt from horizontal (90=nadir).
    roll_deg         : Camera roll (degrees).
    f_mm             : Focal length in millimetres.
    sensor_width_mm  : Physical sensor width in millimetres.

    Returns
    -------
    (lon, lat) tuple, or None if the ray does not intersect the ground plane.
    """
    agl = alt_cam - alt_ground
    if agl < 0.5:
        agl = 0.5  # Safety floor to avoid division by zero

    # Physical pixel size in mm
    px_size_mm = sensor_width_mm / img_w

    # Principal-point offset (centre of image)
    cx, cy = img_w / 2.0, img_h / 2.0

    # Camera-frame coordinates in mm (pinhole model)
    x_cam = (u - cx) * px_size_mm
    y_cam = (v - cy) * px_size_mm
    z_cam = f_mm

    # Normalised direction vector in camera frame
    d_cam = np.array([x_cam, y_cam, z_cam], dtype=float)
    norm = np.linalg.norm(d_cam)
    if norm < 1e-9:
        return None
    d_cam /= norm

    # Note on pitch convention in aerial datasets:
    # If pitch > 45 deg, it frequently represents angle from nadir (e.g. 78 deg = 12 deg tilt from horizontal)
    # or zenith angle. Standardize pitch so pitch_eff is tilt from horizontal (90=nadir, 0=horizontal):
    pitch_eff = pitch_deg
    if pitch_deg > 60.0:
        # Check if this is an oblique perspective where pitch is measured from nadir:
        # In DJI / SfM oblique flights, pitch values around 60-85 correspond to 90 - angle from horizontal
        # Allow testing both or using the user-specified angle.
        pass

    # Rotate to world ENU frame
    R = _rotation_matrix(heading_deg, pitch_deg, roll_deg)
    d_enu = R @ d_cam  # [East, North, Up]

    # Camera position in local ENU (origin = directly below drone on ground)
    cam_enu = np.array([0.0, 0.0, agl])

    # Intersect ray with ground plane (z_ENU = 0)
    # 0 = agl + t * d_enu[2]  →  t = -agl / d_enu[2]
    if abs(d_enu[2]) < 1e-6:
        return None  # Ray nearly parallel to ground
    t = -agl / d_enu[2]
    if t < 0:
        return None  # Ray goes upward

    ground = cam_enu + t * d_enu  # [east_m, north_m, ~0]
    return enu_offset_to_latlon(lon_cam, lat_cam, ground[0], ground[1])


# ---------------------------------------------------------------------------
# GeoDataFrame / KD-tree helpers (used by classifier.py)
# ---------------------------------------------------------------------------

def latlon_to_enu_array(lons: np.ndarray, lats: np.ndarray,
                         ref_lon: float, ref_lat: float) -> np.ndarray:
    """
    Convert arrays of (lon, lat) to local ENU (east_m, north_m) relative to
    a reference point.  Returns shape (N, 2).
    """
    R = 6_371_000.0
    ref_lat_r = radians(ref_lat)
    dlat = np.radians(lats - ref_lat)
    dlon = np.radians(lons - ref_lon)
    north = dlat * R
    east = dlon * R * cos(ref_lat_r)
    return np.column_stack([east, north])
