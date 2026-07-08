#!/usr/bin/env python3

"""test_inference.py -- tests the detection pipeline maths and inference helpers.

Covers the pure tiling/stitching/deduplication utilities and the parts of the
inference engine that can be exercised without loading real model weights
(device resolution, preprocessing, and raw-output postprocessing).
"""

from __future__ import annotations

import numpy as np

from app.inference import ModelInference
from app.utils import (
    calculate_image_statistics,
    crop_and_tile_image,
    decode_base64_image,
    draw_boxes,
    encode_image_base64,
    remove_duplicate_detections,
    stitch_results,
)


def _det(x1: float, y1: float, x2: float, y2: float, conf: float, cls: int = 0) -> dict:
    """Build a detection dict for tests.

    Args:
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.
        conf: Confidence.
        cls: Class id.

    Returns:
        A detection dict.
    """
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf,
            "class_id": cls, "class_name": "larvae"}


def test_tiling_covers_every_pixel() -> None:
    """Tiling with overlap covers the whole image including the far edges."""
    image = np.zeros((1000, 1500, 3), np.uint8)
    tiles, origins = crop_and_tile_image(image, 640, 64)
    cover = np.zeros((1000, 1500), bool)
    for (x, y), tile in zip(origins, tiles):
        h, w = tile.shape[:2]
        cover[y : y + h, x : x + w] = True
    assert cover.all()
    assert max(x for x, _ in origins) == 1500 - 640
    assert max(y for _, y in origins) == 1000 - 640


def test_small_image_is_one_tile() -> None:
    """An image smaller than a tile yields a single full-image tile."""
    tiles, origins = crop_and_tile_image(np.zeros((300, 400, 3), np.uint8), 640, 64)
    assert len(tiles) == 1
    assert origins == [(0, 0)]


def test_stitch_offsets_by_tile_origin() -> None:
    """Stitching shifts tile-local coordinates by their tile origin."""
    stitched = stitch_results([[_det(10, 20, 50, 60, 0.9)]], [(100, 200)], (1500, 1000))
    assert stitched[0]["x1"] == 110 and stitched[0]["y1"] == 220


def test_deduplication_merges_overlapping_boxes() -> None:
    """Highly overlapping same-class boxes collapse to the highest-confidence one."""
    detections = [
        _det(100, 100, 140, 140, 0.9),
        _det(102, 101, 141, 139, 0.7),
        _det(500, 500, 540, 540, 0.6),
    ]
    kept = remove_duplicate_detections(detections, 0.5)
    assert len(kept) == 2
    assert kept[0]["confidence"] == 0.9


def test_statistics_bins_confidence() -> None:
    """Statistics report the count, mean confidence, and ten-band histogram."""
    stats = calculate_image_statistics([_det(0, 0, 1, 1, 0.65), _det(0, 0, 1, 1, 0.95)])
    assert stats["count"] == 2
    assert abs(stats["average_confidence"] - 0.80) < 1e-9
    assert len(stats["confidence_distribution"]) == 10
    assert stats["confidence_distribution"][6] == 1
    assert stats["confidence_distribution"][9] == 1


def test_base64_round_trip_preserves_shape() -> None:
    """A base64-encoded image decodes back to the same shape."""
    image = np.full((12, 16, 3), 100, np.uint8)
    decoded = decode_base64_image(encode_image_base64(image))
    assert decoded.shape == (12, 16, 3)


def test_draw_boxes_does_not_mutate_input() -> None:
    """Drawing boxes returns a new array and leaves the original unchanged."""
    image = np.zeros((50, 50, 3), np.uint8)
    annotated = draw_boxes(image, [_det(5, 5, 20, 20, 0.9)])
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
    assert image.sum() == 0


def test_resolve_device_handles_explicit_and_auto() -> None:
    """Device resolution passes explicit devices through and resolves 'auto'."""
    assert ModelInference._resolve_device("cpu") == "cpu"
    assert ModelInference._resolve_device("auto") in {"cpu", "cuda:0"}


def test_postprocess_converts_raw_predictions() -> None:
    """Raw model predictions convert into detection dicts with class names."""
    engine = object.__new__(ModelInference)

    class _FakeResults:
        """Minimal stand-in for the YOLOv5 results object."""

        pred = [np.array([[10.0, 20.0, 30.0, 40.0, 0.88, 0.0]])]
        names = {0: "larvae"}

    detections = engine.postprocess_results(_FakeResults())
    assert len(detections) == 1
    assert detections[0]["class_name"] == "larvae"
    assert detections[0]["x2"] == 30.0
    assert abs(detections[0]["confidence"] - 0.88) < 1e-6


def test_preprocess_rejects_non_three_channel() -> None:
    """Preprocessing rejects images that are not 3-channel."""
    engine = object.__new__(ModelInference)
    grayscale = np.zeros((10, 10), np.uint8)
    try:
        engine.preprocess_image(grayscale)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
