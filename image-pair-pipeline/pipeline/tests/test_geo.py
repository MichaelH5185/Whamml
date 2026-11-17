"""
tests/test_geo.py
-----------------
Unit tests for the geo-projection utilities.
"""
import math
import pytest
from pipeline.geo import pixel_to_lonlat, haversine_m, enu_offset_to_latlon


class TestHaversine:

    def test_same_point_is_zero(self):
        assert haversine_m(-90.0, 38.0, -90.0, 38.0) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        # ~111,320 m per degree latitude at equator
        d = haversine_m(0.0, 0.0, 0.0, 1.0)
        assert d == pytest.approx(111_195, rel=0.01)

    def test_symmetric(self):
        d1 = haversine_m(-90.0, 38.0, -90.1, 38.1)
        d2 = haversine_m(-90.1, 38.1, -90.0, 38.0)
        assert d1 == pytest.approx(d2, rel=1e-9)


class TestENUOffset:

    def test_north_100m(self):
        lon0, lat0 = -90.0, 38.0
        lon1, lat1 = enu_offset_to_latlon(lon0, lat0, 0.0, 100.0)
        d = haversine_m(lon0, lat0, lon1, lat1)
        assert d == pytest.approx(100.0, rel=0.01)
        assert lat1 > lat0  # moved north

    def test_east_100m(self):
        lon0, lat0 = -90.0, 38.0
        lon1, lat1 = enu_offset_to_latlon(lon0, lat0, 100.0, 0.0)
        d = haversine_m(lon0, lat0, lon1, lat1)
        assert d == pytest.approx(100.0, rel=0.01)
        assert lon1 > lon0  # moved east


class TestPixelToLonLat:

    def test_nadir_centre_pixel_is_camera_position(self):
        """Centre pixel of a perfectly nadir image should project to camera lon/lat."""
        result = pixel_to_lonlat(
            u=960, v=540,
            img_w=1920, img_h=1080,
            lon_cam=-90.0, lat_cam=38.0,
            alt_cam=150.0, alt_ground=0.0,
            heading_deg=0.0,
            pitch_deg=90.0,  # nadir
            roll_deg=0.0,
            f_mm=22.0,
            sensor_width_mm=13.2,
        )
        assert result is not None
        lon, lat = result
        assert lon == pytest.approx(-90.0, abs=1e-5)
        assert lat == pytest.approx(38.0,  abs=1e-5)

    def test_nadir_offset_pixel_moves_east(self):
        """A pixel to the right of centre (nadir camera heading north) → East."""
        result_centre = pixel_to_lonlat(
            u=960, v=540, img_w=1920, img_h=1080,
            lon_cam=-90.0, lat_cam=38.0,
            alt_cam=100.0, alt_ground=0.0,
            heading_deg=0.0, pitch_deg=90.0, roll_deg=0.0,
            f_mm=22.0, sensor_width_mm=13.2,
        )
        result_right = pixel_to_lonlat(
            u=1440, v=540, img_w=1920, img_h=1080,  # 480px to the right
            lon_cam=-90.0, lat_cam=38.0,
            alt_cam=100.0, alt_ground=0.0,
            heading_deg=0.0, pitch_deg=90.0, roll_deg=0.0,
            f_mm=22.0, sensor_width_mm=13.2,
        )
        assert result_right is not None
        assert result_right[0] > result_centre[0]  # lon increases → East

    def test_parallel_ray_returns_none(self):
        """A horizontal camera (pitch=0) looking straight along heading can't hit ground."""
        result = pixel_to_lonlat(
            u=960, v=540, img_w=1920, img_h=1080,
            lon_cam=-90.0, lat_cam=38.0,
            alt_cam=100.0, alt_ground=0.0,
            heading_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
            f_mm=22.0, sensor_width_mm=13.2,
        )
        # Centre pixel of a horizontal camera → parallel to ground → None
        assert result is None

    def test_agl_below_zero_uses_safety_floor(self):
        """Camera below ground altitude should not crash."""
        result = pixel_to_lonlat(
            u=960, v=540, img_w=1920, img_h=1080,
            lon_cam=-90.0, lat_cam=38.0,
            alt_cam=100.0, alt_ground=200.0,  # AGL would be -100 → clamped to 0.5
            heading_deg=0.0, pitch_deg=90.0, roll_deg=0.0,
            f_mm=22.0, sensor_width_mm=13.2,
        )
        # Should not raise; result may be unreliable but should return a value
        assert result is not None
