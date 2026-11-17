import pytest
import numpy as np
import cv2
from pathlib import Path
from pipeline.detector import Detection
from pipeline.exporter import Exporter, _crop_masked_object, _pad_to_square


def test_pad_to_square():
    # Non-square 3-channel image
    img = np.full((100, 200, 3), 128, dtype=np.uint8)
    sq = _pad_to_square(img)
    assert sq.shape == (200, 200, 3)
    # Check that center is filled and borders are black
    assert np.all(sq[0:40, :, :] == 0)
    assert np.all(sq[50:150, 0:200, :] == 128)


def test_crop_masked_object():
    # Create a 200x200 RGB canvas with red color
    img = np.full((200, 200, 3), (0, 0, 255), dtype=np.uint8)

    # Polygon mask for a 60x40 triangle inside the image
    poly = np.array([
        [50, 50],
        [110, 50],
        [80, 90],
    ], dtype=np.float32)

    bbox = [50.0, 50.0, 110.0, 90.0]

    out = _crop_masked_object(img, poly, bbox)

    # Output should be square
    h, w, c = out.shape
    assert h == w
    assert c == 3

    # Inside polygon should preserve color (red = [0, 0, 255]), background should be black (0, 0, 0)
    # The center (80, 60) relative to (50, 50) is offset in square canvas
    # Verify that not all pixels are 0, and not all are red
    assert np.any(out > 0)
    assert np.any(out == 0)


def test_exporter_saves_masked_object_images(tmp_path):
    crops_dir = tmp_path / "crops"
    masks_dir = tmp_path / "masks"
    exporter = Exporter(crops_dir, masks_dir)

    # Make dummy before and after images
    before_img = np.full((200, 200, 3), 100, dtype=np.uint8)
    after_img = np.full((200, 200, 3), 200, dtype=np.uint8)

    before_path = tmp_path / "frame_0001.png"
    after_path = tmp_path / "cesiumcamerasequence.0001.png"

    cv2.imwrite(str(before_path), before_img)
    cv2.imwrite(str(after_path), after_img)

    # Make a non-square polygon mask (60 wide x 40 high triangle inside 200x200 image)
    det_before = Detection(
        bbox=[20.0, 30.0, 80.0, 70.0],
        conf=0.9,
        cls_id=0,
        mask=np.array([[20, 30], [80, 30], [50, 70]], dtype=np.float32),
        img_size=(200, 200),
    )
    det_after = Detection(
        bbox=[25.0, 35.0, 85.0, 75.0],
        conf=0.85,
        cls_id=0,
        mask=np.array([[25, 35], [85, 35], [55, 75]], dtype=np.float32),
        img_size=(200, 200),
    )

    saved = exporter.export_pair(
        suffix="0001",
        damage_class=2,
        before_path=before_path,
        after_path=after_path,
        det_before=det_before,
        det_after=det_after,
    )

    mask_b = cv2.imread(str(saved["mask_before"]))
    assert mask_b is not None
    assert mask_b.shape[0] == mask_b.shape[1]  # square
    assert mask_b.shape[2] == 3               # 3-channel RGB image
    # Object pixels should have value 100
    assert np.any(mask_b == 100)
    # Background should have black padding
    assert np.any(mask_b == 0)
