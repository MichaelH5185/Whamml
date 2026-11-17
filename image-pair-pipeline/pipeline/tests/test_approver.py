"""
tests/test_approver.py
----------------------
Unit tests for the approval preview renderer.
"""
from pathlib import Path
import numpy as np
import cv2
import pytest

from pipeline.detector import Detection
from pipeline.matcher import DetectionPair
from pipeline.approver import render_approval_image, PAIR_PALETTE, parse_index_selection


def make_det(x1, y1, x2, y2, conf=0.9, cls_id=0):
    return Detection(bbox=[x1, y1, x2, y2], mask=None, conf=conf, cls_id=cls_id)


class TestRenderApprovalImage:

    @pytest.fixture
    def dummy_images(self, tmp_path):
        before = tmp_path / "frame_0001.png"
        after  = tmp_path / "cesiumcamerasequence.0001.png"

        # Create 100x100 dummy images
        img_b = np.full((100, 100, 3), 40, dtype=np.uint8)
        img_a = np.full((100, 100, 3), 60, dtype=np.uint8)
        cv2.imwrite(str(before), img_b)
        cv2.imwrite(str(after),  img_a)

        return before, after

    def test_render_with_no_pairs(self, dummy_images):
        before, after = dummy_images
        preview = render_approval_image(
            before_path=before,
            after_path=after,
            proposed_pairs=[],
            dets_before=[make_det(10, 10, 40, 40)],
            dets_after=[make_det(15, 15, 45, 45)],
            suffix="0001",
            target_h=200,
        )
        assert preview is not None
        assert preview.shape[0] == 200 + 54  # 200 + max(54, 11) header
        assert preview.shape[2] == 3

    def test_render_with_multiple_indexed_pairs(self, dummy_images):
        before, after = dummy_images

        det_b1 = make_det(10, 10, 40, 40)
        det_a1 = make_det(12, 12, 42, 42)
        pair1 = DetectionPair("0001", before, after, det_b1, det_a1, iou=0.85)
        pair1.damage_class = 3

        det_b2 = make_det(50, 50, 80, 80)
        det_a2 = make_det(52, 52, 82, 82)
        pair2 = DetectionPair("0001", before, after, det_b2, det_a2, iou=0.78)
        pair2.damage_class = 1

        preview = render_approval_image(
            before_path=before,
            after_path=after,
            proposed_pairs=[pair1, pair2],
            dets_before=[det_b1, det_b2],
            dets_after=[det_a1, det_a2],
            suffix="0001",
            target_h=300,
        )
        assert preview is not None
        assert preview.shape[0] == 300 + 54  # 300 + max(54, 17) header
        assert preview.shape[2] == 3

    def test_palette_distinct_colors(self):
        assert len(PAIR_PALETTE) >= 10
        # All colors should have 3 channels (BGR)
        for c in PAIR_PALETTE:
            assert len(c) == 3
            assert all(0 <= val <= 255 for val in c)


class TestParseIndexSelection:

    def test_single_index(self):
        assert parse_index_selection("1", max_index=5) == [1]
        assert parse_index_selection("3", max_index=5) == [3]

    def test_comma_separated(self):
        assert parse_index_selection("1, 3", max_index=5) == [1, 3]
        assert parse_index_selection("1,2,5", max_index=5) == [1, 2, 5]

    def test_space_separated(self):
        assert parse_index_selection("1 3 4", max_index=5) == [1, 3, 4]

    def test_hash_prefixes(self):
        assert parse_index_selection("#1, #3", max_index=5) == [1, 3]
        assert parse_index_selection("#2 #4", max_index=5) == [2, 4]

    def test_range_syntax(self):
        assert parse_index_selection("1-3", max_index=5) == [1, 2, 3]
        assert parse_index_selection("1-3, 5", max_index=5) == [1, 2, 3, 5]
        assert parse_index_selection("2-4", max_index=5) == [2, 3, 4]

    def test_out_of_range_returns_none(self):
        assert parse_index_selection("0", max_index=5) is None
        assert parse_index_selection("6", max_index=5) is None
        assert parse_index_selection("1, 6", max_index=5) is None
        assert parse_index_selection("3-7", max_index=5) is None

    def test_invalid_syntax_returns_none(self):
        assert parse_index_selection("foo", max_index=5) is None
        assert parse_index_selection("", max_index=5) is None
        assert parse_index_selection("   ", max_index=5) is None
        assert parse_index_selection("3-1", max_index=5) is None
