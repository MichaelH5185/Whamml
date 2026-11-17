"""
tests/test_classifier.py
------------------------
Unit tests for the BuildingIndex class and rating parsing.
"""
import pytest
import tempfile
import os
import pandas as pd

from pipeline.classifier import _parse_rating, BuildingIndex


class TestParseRating:

    def test_known_ratings(self):
        assert _parse_rating("Negligible to Slight") == 1
        assert _parse_rating("negligible to slight") == 1
        assert _parse_rating("Moderate") == 2
        assert _parse_rating("Substantial to Heavy") == 3
        assert _parse_rating("Heavy") == 3
        assert _parse_rating("Destruction") == 5
        assert _parse_rating("Complete") == 5

    def test_numeric_string(self):
        assert _parse_rating("3") == 3
        assert _parse_rating("1") == 1
        assert _parse_rating("5") == 5

    def test_out_of_range_numeric(self):
        assert _parse_rating("9") is None
        assert _parse_rating("0") is None

    def test_unknown_returns_none(self):
        assert _parse_rating("foobar") is None
        assert _parse_rating("") is None


class TestBuildingIndex:

    @pytest.fixture
    def csv_path(self, tmp_path):
        """Write a minimal building CSV and return its path."""
        data = pd.DataFrame({
            "HANDLE":   [1001, 1002, 1003],
            "Lon":      [-90.234183, -90.240221, -90.224793],
            "Lat":      [38.670556,  38.669530,  38.672439],
            "Rating":   ["Substantial to Heavy", "Moderate", "Negligible to Slight"],
            "SITEADDR": ["4300 SAN FRANCISCO AV", "3526 N TAYLOR AV", "4217 FAIR AV"],
        })
        p = tmp_path / "buildings.csv"
        data.to_csv(p, index=False)
        return str(p)

    def test_exact_match(self, csv_path):
        idx = BuildingIndex(csv_path, match_radius_m=50.0)
        # Query exactly at building 1 location → class 3
        cls = idx.query(-90.234183, 38.670556)
        assert cls == 3

    def test_query_details_returns_address_and_distance(self, csv_path):
        idx = BuildingIndex(csv_path, match_radius_m=50.0)
        details = idx.query_details(-90.234183, 38.670556)
        assert details is not None
        assert details["class"] == 3
        assert details["address"] == "4300 SAN FRANCISCO AV"
        assert details["distance_m"] < 1.0

    def test_nearest_building_always_returns_closest(self, csv_path):
        idx = BuildingIndex(csv_path, match_radius_m=10.0)
        # 500m away: query_details is None, but nearest_building still gives info
        assert idx.query_details(-90.0, 38.0) is None
        nearest = idx.nearest_building(-90.0, 38.0)
        assert nearest["distance_m"] > 1000.0
        assert nearest["address"] != ""

    def test_nearby_match(self, csv_path):
        # Offset by ~5 m from building 2 → class 2
        idx = BuildingIndex(csv_path, match_radius_m=50.0)
        # 0.00005° lat ≈ 5.5 m
        cls = idx.query(-90.240221, 38.669530 + 0.00005)
        assert cls == 2

    def test_too_far_returns_none(self, csv_path):
        idx = BuildingIndex(csv_path, match_radius_m=10.0)
        # Query 500 m away from all buildings
        cls = idx.query(-90.0, 38.0)
        assert cls is None


class TestDedup:
    """Test that deduplication caps pairs per building."""

    def test_max_per_building(self):
        from pipeline.dedup import deduplicate
        from pipeline.matcher import DetectionPair
        from pipeline.detector import Detection
        from pathlib import Path

        def fake_pair(lon, lat, iou):
            det = Detection(bbox=[0,0,50,50], mask=None, conf=0.9, cls_id=0)
            p = DetectionPair("0001", Path("b.jpg"), Path("a.jpg"), det, det, iou)
            p.damage_class = 1
            p.building_lon = lon
            p.building_lat = lat
            return p

        # 10 pairs for the same building
        pairs = [fake_pair(-90.234183, 38.670556, 0.9 - i * 0.05) for i in range(10)]
        result = deduplicate(pairs, max_per_building=3)
        assert len(result) == 3
        # Should keep the top 3 by IoU
        assert all(p.iou >= 0.9 - 2 * 0.05 for p in result)


class TestPointInMaskJson:
    """Test loading per-frame JSON and matching detections to points."""

    def test_load_frame_buildings_json(self, tmp_path):
        import json
        from pipeline.classifier import load_frame_buildings_json

        data = {
            "frame_0001.jpg": [
                {
                    "handle": "13765050175",
                    "address": "1010 N KINGSHIGHWAY BLVD",
                    "damage_rating": "Moderate",
                    "pixel_x": 768.32,
                    "pixel_y": 496.01,
                    "distance_meters": 110.36
                }
            ]
        }
        json_file = tmp_path / "frame_bldgs.json"
        with open(json_file, "w") as f:
            json.dump(data, f)

        res = load_frame_buildings_json(str(json_file))
        # Should be accessible via exact key 'frame_0001.jpg', stem 'frame_0001', suffix '0001'
        assert "frame_0001.jpg" in res
        assert "frame_0001" in res
        assert "0001" in res
        assert len(res["frame_0001.jpg"]) == 1
        assert res["frame_0001.jpg"][0]["handle"] == "13765050175"

    def test_load_frame_buildings_json_nested_structure(self, tmp_path):
        import json
        from pipeline.classifier import load_frame_buildings_json

        data = {
            "metadata": {"sequence": "test", "total_frames": 1},
            "frames": {
                "frame_0002.jpg": [
                    {
                        "handle": "13765050176",
                        "address": "1012 N KINGSHIGHWAY BLVD",
                        "damage_rating": "Severe",
                        "pixel_x": 500.0,
                        "pixel_y": 400.0,
                        "distance_meters": 90.0
                    }
                ]
            }
        }
        json_file = tmp_path / "nested_bldgs.json"
        with open(json_file, "w") as f:
            json.dump(data, f)

        res = load_frame_buildings_json(str(json_file))
        assert "frame_0002.jpg" in res
        assert "frame_0002" in res
        assert "0002" in res
        assert len(res["frame_0002.jpg"]) == 1
        assert res["frame_0002.jpg"][0]["handle"] == "13765050176"

    def test_match_detection_to_point_inside_polygon(self):
        import numpy as np
        from pipeline.detector import Detection
        from pipeline.classifier import match_detection_to_point

        # Triangle mask from (100, 100), (300, 100), (200, 300)
        # Center of bbox is (200, 200)
        poly = np.array([
            [100, 100],
            [300, 100],
            [200, 300]
        ], dtype=np.float32)

        det = Detection(
            bbox=[100.0, 100.0, 300.0, 300.0],
            mask=poly,
            conf=0.9,
            cls_id=0,
            img_size=(1920, 1080),
        )

        candidates = [
            {
                "handle": "outside_pt",
                "address": "Outside St",
                "damage_rating": "Destruction",
                "pixel_x": 110.0,
                "pixel_y": 250.0,  # inside bbox but outside triangle
                "distance_meters": 10.0,
            },
            {
                "handle": "closest_camera_pt",
                "address": "Close Drone Ave",
                "damage_rating": "Moderate",
                "pixel_x": 200.0,
                "pixel_y": 250.0,  # inside triangle, distance_meters = 45.0m
                "distance_meters": 45.0,
            },
            {
                "handle": "far_camera_central_pt",
                "address": "Far Away Central Blvd",
                "damage_rating": "Substantial to Heavy",
                "pixel_x": 200.0,
                "pixel_y": 180.0,  # inside triangle, central to mask, but distance_meters = 250.0m
                "distance_meters": 250.0,
            }
        ]

        match = match_detection_to_point(det, candidates)
        assert match is not None
        # Preference given to the point with closest distance_meters
        assert match["handle"] == "closest_camera_pt"
        assert match["class"] == 2
        assert match["address"] == "Close Drone Ave"
        assert match["distance_meters"] == 45.0

    def test_match_detection_to_point_none_inside(self):
        import numpy as np
        from pipeline.detector import Detection
        from pipeline.classifier import match_detection_to_point

        poly = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
        det = Detection(bbox=[10.0, 10.0, 50.0, 50.0], mask=poly, conf=0.9, cls_id=0)

        candidates = [
            {
                "handle": "far_away",
                "address": "Far Away Rd",
                "damage_rating": "Moderate",
                "pixel_x": 500.0,
                "pixel_y": 500.0,
            }
        ]
        match = match_detection_to_point(det, candidates)
        assert match is None
