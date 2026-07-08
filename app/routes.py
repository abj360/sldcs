#!/usr/bin/env python3

"""routes.py -- handles upload validation and detection requests.

Contains the functions that validate uploads and run the detection pipeline to
build API responses. These handlers own request validation and response
assembly; they delegate all model work to :class:`app.inference.YOLOv5Inference`
and all image maths to :mod:`app.utils`, and they never import torch directly.
"""

from __future__ import annotations

import time
from typing import Final

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.config import Settings
from app.inference import YOLOv5Inference
from app.models import (
    BatchDetectionResponse,
    BatchImageCount,
    DetectionBox,
    DetectionResult,
)
from app.utils import calculate_image_statistics, draw_boxes, encode_image_base64

ALLOWED_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {"image/jpeg", "image/png", "image/bmp", "image/x-ms-bmp"}
)
ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

# A small compressed file can decode to an enormous bitmap ("decompression
# bomb"); the byte-size limit alone does not bound the decoded pixel count, so
# reject images whose decoded area is implausibly large before the pipeline
# spends memory tiling, annotating, and re-encoding them. 120 MP comfortably
# clears any real full-tray specimen photograph.
MAX_IMAGE_PIXELS: Final[int] = 120_000_000

# Upper bound on images accepted in a single request, so one call cannot pin the
# instrument by submitting an unbounded number of files.
MAX_UPLOAD_FILES: Final[int] = 64


def _decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR array.

    Args:
        data: Raw encoded image bytes.

    Returns:
        The decoded ``(H, W, 3)`` BGR image.

    Raises:
        HTTPException: 400 if the bytes cannot be decoded as an image, or 413 if
            the decoded image exceeds :data:`MAX_IMAGE_PIXELS`.
    """
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image; the file may be corrupt.",
        )
    height, width = image.shape[:2]
    if height * width > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Image resolution {width}×{height} exceeds the "
                f"{MAX_IMAGE_PIXELS:,}-pixel limit — resize and try again."
            ),
        )
    return image


def enforce_upload_count(count: int) -> None:
    """Reject requests that submit more images than the per-request limit.

    Args:
        count: Number of files in the request.

    Raises:
        HTTPException: 413 if ``count`` exceeds :data:`MAX_UPLOAD_FILES`.
    """
    if count > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Too many files in one request — send at most {MAX_UPLOAD_FILES}.",
        )


async def validate_upload(upload: UploadFile, max_file_size: int) -> bytes:
    """Validate an uploaded file's type and size and return its bytes.

    Args:
        upload: The incoming multipart file.
        max_file_size: Maximum accepted size in bytes.

    Returns:
        The full file contents as bytes.

    Raises:
        HTTPException: 415 for an unsupported type, 413 if it exceeds the size
            limit, 400 if the file is empty.
    """
    extension = ("." + upload.filename.rsplit(".", 1)[-1].lower()) if upload.filename and "." in upload.filename else ""
    content_type = (upload.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES and extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file — use JPG, PNG, or BMP.",
        )
    data = await upload.read()
    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )
    if len(data) > max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {max_file_size} bytes — resize and try again.",
        )
    return data


def _detections_to_boxes(detections: list[dict]) -> list[DetectionBox]:
    """Convert pipeline detection dicts into :class:`DetectionBox` models.

    Args:
        detections: Detection dicts in original-image coordinates.

    Returns:
        ``DetectionBox`` instances with stable 1-based ids.
    """
    return [
        DetectionBox(
            id=index,
            x1=detection["x1"],
            y1=detection["y1"],
            x2=detection["x2"],
            y2=detection["y2"],
            confidence=detection["confidence"],
            class_id=detection["class_id"],
            class_name=detection["class_name"],
        )
        for index, detection in enumerate(detections, start=1)
    ]


def _build_detection_result(
    filename: str | None,
    image: np.ndarray,
    inference: YOLOv5Inference,
    encode_annotated: bool,
) -> DetectionResult:
    """Run the pipeline on one image and assemble its :class:`DetectionResult`.

    Args:
        filename: Original filename, if known.
        image: Decoded BGR image.
        inference: The shared detection engine.
        encode_annotated: Whether to draw boxes and include the base64 image.

    Returns:
        The fully populated detection result for this image.
    """
    start = time.perf_counter()
    outcome = inference.detect(image)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    detections = outcome["detections"]
    stats = calculate_image_statistics(detections)
    height, width = image.shape[:2]

    annotated_image = None
    if encode_annotated:
        annotated_image = encode_image_base64(draw_boxes(image, detections))

    return DetectionResult(
        filename=filename,
        larvae_count=len(detections),
        detections=_detections_to_boxes(detections),
        average_confidence=stats["average_confidence"],
        processing_time_ms=elapsed_ms,
        tiles_scanned=outcome["tiles_scanned"],
        duplicates_merged=outcome["duplicates_merged"],
        image_width=width,
        image_height=height,
        confidence_distribution=stats["confidence_distribution"],
        annotated_image=annotated_image,
    )


def handle_detection(
    images: list[tuple[str | None, np.ndarray]],
    inference: YOLOv5Inference,
    settings: Settings,
) -> list[DetectionResult]:
    """Run the full five-stage pipeline on one or more images, with annotations.

    Args:
        images: ``(filename, image)`` pairs to process.
        inference: The shared detection engine.
        settings: Active application settings (reserved for per-request tuning).

    Returns:
        One annotated :class:`DetectionResult` per input image, in order.
    """
    del settings  # Thresholds are already applied on the shared engine.
    return [
        _build_detection_result(filename, image, inference, encode_annotated=True)
        for filename, image in images
    ]


def handle_batch_detection(
    images: list[tuple[str | None, np.ndarray]],
    inference: YOLOv5Inference,
    settings: Settings,
) -> BatchDetectionResponse:
    """Run the pipeline on a batch, returning counts only (no image encoding).

    Skipping annotation and base64 encoding keeps batch throughput high.

    Args:
        images: ``(filename, image)`` pairs to process.
        inference: The shared detection engine.
        settings: Active application settings (reserved for per-request tuning).

    Returns:
        The aggregate :class:`BatchDetectionResponse`.
    """
    del settings
    per_image: list[BatchImageCount] = []
    for filename, image in images:
        result = _build_detection_result(filename, image, inference, encode_annotated=False)
        per_image.append(
            BatchImageCount(
                filename=filename or "",
                larvae_count=result.larvae_count,
                average_confidence=result.average_confidence,
            )
        )
    total_larvae = sum(item.larvae_count for item in per_image)
    average = total_larvae / len(per_image) if per_image else 0.0
    return BatchDetectionResponse(
        image_count=len(per_image),
        total_larvae=total_larvae,
        average_per_image=average,
        results=per_image,
    )
