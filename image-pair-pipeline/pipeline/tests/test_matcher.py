"""
tests/test_matcher.py
---------------------
Unit tests for size-normalised IoU, UUID extraction, and Hungarian matching.
"""
import pytest
from pipeline.detector import Detection
from pipeline.matcher import (
    calculate_iou, size_normalized_iou, match_detections,
    scan_before_dir, scan_after_dir, paired_suffixes,
    _BEFORE_PATTERN, _AFTER_PATTERN, _suffix,
)
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper to build a Detection quickly
# ---------------------------------------------------------------------------

def make_det(x1, y1, x2, y2, conf=0.9, cls_id=0):
    return Detection(bbox=[x1, y1, x2, y2], mask=None, conf=conf, cls_id=cls_id)


# ---------------------------------------------------------------------------
# IoU tests
# ---------------------------------------------------------------------------

class TestSizePenalizedIoU:

    def test_identical_boxes(self):
        a = make_det(0, 0, 100, 100)
        b = make_det(0, 0, 100, 100)
        assert calculate_iou(a, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = make_det(0,   0, 50, 50)
        b = make_det(60, 60, 110, 110)
        assert calculate_iou(a, b) == 0.0

    def test_half_overlap_equal_size(self):
        a = make_det(0,  0, 100, 100)
        b = make_det(50, 0, 150, 100)
        # intersection = 50×100 = 5000, union = 15000 → 1/3
        assert calculate_iou(a, b) == pytest.approx(1.0 / 3.0)

    def test_contained_small_in_large_punished(self):
        large = make_det(0, 0, 200, 200)   # area = 40000
        small = make_det(50, 50, 100, 100) # area = 2500
        # IoU = 2500 / 40000 = 0.0625
        # size_ratio = 2500 / 40000 = 0.0625
        # With default size penalty weight=1: score = 0.0625 * 0.0625 = ~0.0039
        # Crucially: it is NOT 1.0!
        score = calculate_iou(large, small, size_penalty_weight=1.0)
        assert score == pytest.approx(0.0625 * 0.0625)
        assert score < 0.01

        # With standard IoU (weight=0): score = 0.0625, still strongly penalizing containment
        score_std = calculate_iou(large, small, size_penalty_weight=0.0)
        assert score_std == pytest.approx(0.0625)

    def test_partial_containment(self):
        a = make_det(0,  0, 100, 100)   # area 10000
        b = make_det(0,  0, 200, 50)    # area 10000, overlap = 100*50=5000
        # union = 15000, size_ratio = 1.0 → 1/3
        assert calculate_iou(a, b) == pytest.approx(1.0 / 3.0)

    def test_zero_area_returns_zero(self):
        a = make_det(10, 10, 10, 10)  # zero area
        b = make_det(0,   0, 50, 50)
        assert calculate_iou(a, b) == 0.0

    def test_different_resolution_images_no_false_match(self):
        # Image A is 4000x3000 (drone). Box is on the left side: x in [800, 1600] -> x_norm in [0.2, 0.4]
        det_a = Detection(
            bbox=[800, 300, 1600, 900], mask=None, conf=0.9, cls_id=0,
            img_size=(4000, 3000),
        )
        # Image B is 1920x1080 (Cesium). Box is on the right side: x in [1200, 1600] -> x_norm in [0.625, 0.833]
        # In raw pixels, [800, 1600] and [1200, 1600] overlap between 1200 and 1600.
        # But in normalized screen space, [0.2, 0.4] and [0.625, 0.833] have ZERO overlap!
        det_b = Detection(
            bbox=[1200, 200, 1600, 500], mask=None, conf=0.9, cls_id=0,
            img_size=(1920, 1080),
        )
        assert calculate_iou(det_a, det_b) == 0.0

    def test_different_resolution_images_true_match(self):
        # Both represent a building located at normalized [0.2, 0.2, 0.4, 0.4]
        det_a = Detection(
            bbox=[800, 600, 1600, 1200], mask=None, conf=0.9, cls_id=0,
            img_size=(4000, 3000),
        )
        det_b = Detection(
            bbox=[384, 216, 768, 432], mask=None, conf=0.9, cls_id=0,
            img_size=(1920, 1080),
        )
        # Since they occupy the exact same relative space, IoU should be 1.0
        assert calculate_iou(det_a, det_b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# UUID extraction tests
# ---------------------------------------------------------------------------

class TestUUIDExtraction:

    def test_before_pattern(self):
        p = Path("frame_0042.png")
        assert _suffix(p, _BEFORE_PATTERN) == "0042"

    def test_before_pattern_case_insensitive(self):
        p = Path("Frame_0007.JPG")
        assert _suffix(p, _BEFORE_PATTERN) == "0007"

    def test_after_pattern(self):
        p = Path("cesiumcamerasequence.0099.jpg")
        assert _suffix(p, _AFTER_PATTERN) == "0099"

    def test_after_pattern_case_insensitive(self):
        p = Path("CesiumCameraSequence.0001.PNG")
        assert _suffix(p, _AFTER_PATTERN) == "0001"

    def test_no_match_returns_none(self):
        p = Path("random_image.jpg")
        assert _suffix(p, _BEFORE_PATTERN) is None
        assert _suffix(p, _AFTER_PATTERN)  is None


# ---------------------------------------------------------------------------
# Frame step / pairing tests
# ---------------------------------------------------------------------------

class TestPairedSuffixes:

    def test_every_5th(self):
        before = {f"{i:04d}": Path(f"frame_{i:04d}.png") for i in range(1, 21)}
        after  = {f"{i:04d}": Path(f"cesiumcamerasequence.{i:04d}.jpg") for i in range(1, 25)}
        result = paired_suffixes(before, after, frame_step=5)
        assert len(result) == 4   # 20 common → every 5th = 4
        assert result[0] == "0001"
        assert result[1] == "0006"

    def test_extra_after_frames_ignored(self):
        before = {"0001": Path("frame_0001.png"), "0002": Path("frame_0002.png")}
        after  = {
            "0001": Path("cesiumcamerasequence.0001.jpg"),
            "0002": Path("cesiumcamerasequence.0002.jpg"),
            "0003": Path("cesiumcamerasequence.0003.jpg"),  # extra
        }
        result = paired_suffixes(before, after, frame_step=1)
        assert set(result) == {"0001", "0002"}

    def test_step_1_returns_all_common(self):
        before = {f"{i:04d}": Path() for i in range(10)}
        after  = {f"{i:04d}": Path() for i in range(10)}
        result = paired_suffixes(before, after, frame_step=1)
        assert len(result) == 10

    def test_start_pair_filtering(self):
        before = {f"{i:04d}": Path() for i in range(20)}
        after  = {f"{i:04d}": Path() for i in range(20)}
        result = paired_suffixes(before, after, frame_step=1, start_suffix="0010")
        assert result[0] == "0010"
        assert len(result) == 10

        result2 = paired_suffixes(before, after, frame_step=1, start_suffix="frame_0015")
        assert result2[0] == "0015"
        assert len(result2) == 5


# ---------------------------------------------------------------------------
# Hungarian matching tests
# ---------------------------------------------------------------------------

class TestMatchDetections:

    def test_single_pair_above_threshold(self):
        a = make_det(0, 0, 100, 100)
        b = make_det(10, 10, 110, 110)  # overlapping
        pairs = match_detections([a], [b], iou_threshold=0.1)
        assert len(pairs) == 1
        assert pairs[0][2] > 0.1

    def test_below_threshold_dropped(self):
        a = make_det(0,   0,  50, 50)
        b = make_det(200, 200, 250, 250)  # no overlap
        pairs = match_detections([a], [b], iou_threshold=0.1)
        assert len(pairs) == 0

    def test_optimal_assignment_chosen(self):
        # Two before detections, two after detections
        # Best assignment: a1↔b1, a2↔b2
        a1 = make_det(0,   0, 100, 100)
        a2 = make_det(200, 0, 300, 100)
        b1 = make_det(5,   5, 105, 105)   # close to a1
        b2 = make_det(205, 5, 305, 105)   # close to a2
        pairs = match_detections([a1, a2], [b1, b2], iou_threshold=0.1)
        assert len(pairs) == 2
        # Each before should be matched to its closest after
        bef_boxes = {tuple(p[0].bbox) for p in pairs}
        aft_boxes  = {tuple(p[1].bbox) for p in pairs}
        assert tuple(a1.bbox) in bef_boxes
        assert tuple(a2.bbox) in bef_boxes

    def test_empty_inputs(self):
        assert match_detections([], [], 0.1) == []
        assert match_detections([make_det(0,0,10,10)], [], 0.1) == []
        assert match_detections([], [make_det(0,0,10,10)], 0.1) == []
