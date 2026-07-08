#!/usr/bin/env python3

"""utils.py -- provides the tiling, stitching, and deduplication image helpers.

This module implements the tiling, stitching, and duplicate-removal stages of the
detection pipeline as pure functions, plus supporting helpers for drawing
annotations, computing summary statistics, and base64 image encoding/decoding.
Nothing here loads a model, performs inference, or touches HTTP; a detection is
represented as a plain dict so these functions stay independent of the inference
engine and the API schemas.

A detection dict has the keys: ``x1``, ``y1``, ``x2``, ``y2`` (pixel corners),
``confidence`` (float), ``class_id`` (int), and ``class_name`` (str).
"""

from __future__ import annotations

import base64
from typing import Any, Final

import cv2
import numpy as np

# SLDCS teal in BGR order (OpenCV convention); source RGB is #2BE1CB.
BOX_COLOR_BGR: Final[tuple[int, int, int]] = (203, 225, 43)
BOX_THICKNESS: Final[int] = 2
LABEL_FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE: Final[float] = 0.4
LABEL_THICKNESS: Final[int] = 1

# Confidence histogram spans [0, 1] in ten fixed 0.1-wide bands.
CONFIDENCE_BAND_COUNT: Final[int] = 10


def _tile_positions(length: int, tile_size: int, step: int) -> list[int]:
    """Compute tile start offsets covering a single axis, including the far edge.

    Args:
        length: Axis length in pixels.
        tile_size: Tile edge length in pixels.
        step: Distance between successive tile starts (``tile_size - overlap``).

    Returns:
        Ascending, de-duplicated start offsets. When ``length`` does not divide
        evenly, the final offset is clamped so the last tile ends exactly at the
        far edge (guaranteeing full coverage).
    """
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def crop_and_tile_image(
    image: np.ndarray, tile_size: int, overlap: int
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Segment an image into overlapping square tiles (pipeline stage 2).

    Args:
        image: Source image as an ``(H, W, C)`` array.
        tile_size: Square tile edge length in pixels.
        overlap: Overlap between adjacent tiles in pixels.

    Returns:
        A tuple ``(tiles, origins)`` where ``tiles`` are the cropped tile arrays
        and ``origins`` are their ``(x, y)`` top-left offsets in the source image.
        Edge tiles may be smaller than ``tile_size`` only when the whole image is
        smaller than a tile.

    Raises:
        ValueError: If ``overlap`` is negative or not smaller than ``tile_size``.
    """
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size.")
    height, width = image.shape[:2]
    step = tile_size - overlap
    tiles: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for y in _tile_positions(height, tile_size, step):
        for x in _tile_positions(width, tile_size, step):
            tile = image[y : y + tile_size, x : x + tile_size]
            tiles.append(tile)
            origins.append((x, y))
    return tiles, origins


def stitch_results(
    tiles_results: list[list[dict]],
    tile_origins: list[tuple[int, int]],
    original_size: tuple[int, int],
) -> list[dict]:
    """Map per-tile detections back into original-image coordinates (stage 4).

    Args:
        tiles_results: One list of tile-local detection dicts per tile, aligned
            with ``tile_origins``.
        tile_origins: The ``(x, y)`` origin of each tile in the original image.
        original_size: The original image ``(width, height)`` used to clamp boxes
            to the image bounds.

    Returns:
        A single flat list of detection dicts in original-image coordinates.

    Raises:
        ValueError: If ``tiles_results`` and ``tile_origins`` differ in length.
    """
    if len(tiles_results) != len(tile_origins):
        raise ValueError("tiles_results and tile_origins must be the same length.")
    width, height = original_size
    stitched: list[dict] = []
    for detections, (origin_x, origin_y) in zip(tiles_results, tile_origins):
        for detection in detections:
            stitched.append(
                {
                    **detection,
                    "x1": float(min(max(detection["x1"] + origin_x, 0), width)),
                    "y1": float(min(max(detection["y1"] + origin_y, 0), height)),
                    "x2": float(min(max(detection["x2"] + origin_x, 0), width)),
                    "y2": float(min(max(detection["y2"] + origin_y, 0), height)),
                }
            )
    return stitched


def _iou(box_a: dict, box_b: dict) -> float:
    """Compute intersection-over-union of two detection boxes.

    Args:
        box_a: First detection dict.
        box_b: Second detection dict.

    Returns:
        The IoU in [0, 1]; 0 when the boxes do not overlap.
    """
    inter_x1 = max(box_a["x1"], box_b["x1"])
    inter_y1 = max(box_a["y1"], box_b["y1"])
    inter_x2 = min(box_a["x2"], box_b["x2"])
    inter_y2 = min(box_a["y2"], box_b["y2"])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, box_a["x2"] - box_a["x1"]) * max(0.0, box_a["y2"] - box_a["y1"])
    area_b = max(0.0, box_b["x2"] - box_b["x1"]) * max(0.0, box_b["y2"] - box_b["y1"])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def remove_duplicate_detections(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Reconcile detections found in more than one overlapping tile (stage 5).

    Applies greedy, class-aware non-maximum suppression: detections are taken in
    descending confidence order, and any later detection of the same class whose
    IoU with an already-kept detection exceeds ``iou_threshold`` is dropped as a
    duplicate.

    Args:
        detections: Stitched detection dicts in original-image coordinates.
        iou_threshold: IoU above which two same-class boxes are the same animal.

    Returns:
        The retained detections, in descending confidence order.
    """
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: list[dict] = []
    for candidate in ordered:
        is_duplicate = any(
            candidate["class_id"] == keeper["class_id"]
            and _iou(candidate, keeper) > iou_threshold
            for keeper in kept
        )
        if not is_duplicate:
            kept.append(candidate)
    return kept


def draw_boxes(
    image: np.ndarray, detections: list[dict], draw_labels: bool = True
) -> np.ndarray:
    """Draw detection boxes (and optional confidence labels) on a copy of an image.

    Args:
        image: Source image as an ``(H, W, C)`` BGR array.
        detections: Detection dicts to draw, in image coordinates.
        draw_labels: Whether to draw a confidence label above each box.

    Returns:
        A new annotated image; the input array is not modified.
    """
    annotated = image.copy()
    for detection in detections:
        pt1 = (int(round(detection["x1"])), int(round(detection["y1"])))
        pt2 = (int(round(detection["x2"])), int(round(detection["y2"])))
        cv2.rectangle(annotated, pt1, pt2, BOX_COLOR_BGR, BOX_THICKNESS)
        if draw_labels:
            label = f"{detection['confidence']:.2f}"
            label_y = max(pt1[1] - 4, 0)
            cv2.putText(
                annotated,
                label,
                (pt1[0], label_y),
                LABEL_FONT,
                LABEL_SCALE,
                BOX_COLOR_BGR,
                LABEL_THICKNESS,
                cv2.LINE_AA,
            )
    return annotated


def calculate_image_statistics(detections: list[dict]) -> dict[str, Any]:
    """Summarize a set of detections for the results panel.

    Args:
        detections: Detection dicts to summarize.

    Returns:
        A dict with ``count`` (int), ``average_confidence`` (float, 0 if none),
        and ``confidence_distribution`` (list of ten ints, one per 0.1 band).
    """
    confidences = [float(d["confidence"]) for d in detections]
    distribution = [0] * CONFIDENCE_BAND_COUNT
    for confidence in confidences:
        band = min(int(confidence * CONFIDENCE_BAND_COUNT), CONFIDENCE_BAND_COUNT - 1)
        distribution[band] += 1
    average = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "count": len(detections),
        "average_confidence": average,
        "confidence_distribution": distribution,
    }


def encode_image_base64(image: np.ndarray) -> str:
    """Encode a BGR image as a base64 PNG string.

    Args:
        image: Image array to encode.

    Returns:
        The base64-encoded PNG bytes as an ASCII string (no data-URI prefix).

    Raises:
        ValueError: If the image cannot be PNG-encoded.
    """
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to PNG-encode image.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_base64_image(encoded: str) -> np.ndarray:
    """Decode a base64 PNG/JPEG string into a BGR image array.

    Args:
        encoded: Base64-encoded image bytes, with or without a data-URI prefix.

    Returns:
        The decoded image as an ``(H, W, C)`` BGR array.

    Raises:
        ValueError: If the string cannot be decoded into an image.
    """
    if "," in encoded and encoded.strip().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    raw = base64.b64decode(encoded)
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode base64 image data.")
    return image
