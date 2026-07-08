#!/usr/bin/env python3

"""models.py -- defines the API request and response schemas.

This module defines the data shapes exchanged over the HTTP API and nothing
else: it contains no business logic, no I/O, and no detection code. Every schema
is a plain, validated data container.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    """A single detected larva within an image.

    Coordinates are absolute pixel positions in the original (stitched) image,
    not tile-local or normalized values.

    Attributes:
        id: Stable 1-based index of this detection within its image.
        x1: Left edge of the bounding box, in pixels.
        y1: Top edge of the bounding box, in pixels.
        x2: Right edge of the bounding box, in pixels.
        y2: Bottom edge of the bounding box, in pixels.
        confidence: Detection confidence in the range [0, 1].
        class_id: Numeric class id assigned by the detector.
        class_name: Human-readable class name.
    """

    id: int = Field(..., ge=1, description="1-based index of the detection within the image.")
    x1: float = Field(..., description="Left edge of the box, in pixels.")
    y1: float = Field(..., description="Top edge of the box, in pixels.")
    x2: float = Field(..., description="Right edge of the box, in pixels.")
    y2: float = Field(..., description="Bottom edge of the box, in pixels.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence [0, 1].")
    class_id: int = Field(..., description="Numeric detector class id.")
    class_name: str = Field(..., description="Human-readable class name.")


class DetectionResult(BaseModel):
    """The full result of running the pipeline on one image.

    Carries the count, every detection, the summary statistics the results
    screen displays, and (for single-image detection) the annotated image
    encoded as a base64 PNG.

    Attributes:
        filename: Original filename of the processed image, if known.
        larvae_count: Number of larvae detected after duplicate removal.
        detections: Every retained detection.
        average_confidence: Mean confidence across detections (0 if none).
        processing_time_ms: Wall-clock pipeline time in milliseconds.
        tiles_scanned: Number of tiles the image was split into and scanned.
        duplicates_merged: Detections removed as cross-tile duplicates.
        image_width: Width of the original image in pixels.
        image_height: Height of the original image in pixels.
        confidence_distribution: Detection counts per 0.1-wide confidence band,
            from [0.0, 0.1) through [0.9, 1.0]; always length 10.
        annotated_image: Base64-encoded PNG of the annotated image, or None when
            image encoding was skipped (e.g. batch mode).
    """

    filename: str | None = Field(None, description="Original filename, if known.")
    larvae_count: int = Field(..., ge=0, description="Larvae detected after deduplication.")
    detections: list[DetectionBox] = Field(default_factory=list, description="Retained detections.")
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean detection confidence.")
    processing_time_ms: float = Field(..., ge=0.0, description="Pipeline wall-clock time (ms).")
    tiles_scanned: int = Field(..., ge=0, description="Number of tiles scanned.")
    duplicates_merged: int = Field(..., ge=0, description="Cross-tile duplicate detections removed.")
    image_width: int = Field(..., gt=0, description="Original image width (px).")
    image_height: int = Field(..., gt=0, description="Original image height (px).")
    confidence_distribution: list[int] = Field(
        default_factory=list, description="Counts per 0.1 confidence band (length 10)."
    )
    annotated_image: str | None = Field(
        None, description="Base64 PNG of the annotated image, or None if not encoded."
    )


class BatchImageCount(BaseModel):
    """The count-only result for a single image within a batch.

    Attributes:
        filename: Original filename of the image.
        larvae_count: Larvae detected after duplicate removal.
        average_confidence: Mean detection confidence for this image.
    """

    filename: str = Field(..., description="Original filename of the image.")
    larvae_count: int = Field(..., ge=0, description="Larvae detected after deduplication.")
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean detection confidence.")


class BatchDetectionRequest(BaseModel):
    """Optional per-request overrides for a batch detection call.

    The images themselves are sent as multipart file uploads; this schema
    documents the tunable options that may accompany them.

    Attributes:
        conf_threshold: Optional confidence-threshold override for this batch.
    """

    conf_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Optional confidence-threshold override."
    )


class BatchDetectionResponse(BaseModel):
    """Aggregate result of running the pipeline over a batch of images.

    Attributes:
        image_count: Number of images processed.
        total_larvae: Sum of larvae detected across all images.
        average_per_image: Mean larvae per image (0 if no images).
        results: Per-image count-only results.
    """

    image_count: int = Field(..., ge=0, description="Number of images processed.")
    total_larvae: int = Field(..., ge=0, description="Total larvae across all images.")
    average_per_image: float = Field(..., ge=0.0, description="Mean larvae per image.")
    results: list[BatchImageCount] = Field(
        default_factory=list, description="Per-image count-only results."
    )


class HealthCheckResponse(BaseModel):
    """Liveness/readiness status of the instrument.

    Attributes:
        status: Overall status string ("ok" or "unavailable").
        model_loaded: Whether the detection model is loaded and ready.
        device: Compute device the model is loaded on.
        version: Application version string.
    """

    status: str = Field(..., description="Overall status: 'ok' or 'unavailable'.")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready.")
    device: str = Field(..., description="Compute device the model runs on.")
    version: str = Field(..., description="Application version string.")


class ModelInfoResponse(BaseModel):
    """Metadata describing the active production model.

    Reproduces the model registry entry and pretrained metadata honestly,
    including whether the model was trained on project data.

    Attributes:
        version: Registry version identifier of the production model.
        source: Provenance description of the model.
        trained_on_project_data: Whether the model was trained on SLDCS data.
        status: Registry status of the model.
        device: Compute device the model is loaded on.
        class_count: Number of classes the model can detect.
        class_names: The model's class names.
        note: Plain-language note about the model's current state.
    """

    version: str = Field(..., description="Registry version identifier.")
    source: str = Field(..., description="Model provenance description.")
    trained_on_project_data: bool = Field(..., description="Trained on SLDCS data?")
    status: str = Field(..., description="Registry status string.")
    device: str = Field(..., description="Compute device the model runs on.")
    class_count: int = Field(..., ge=0, description="Number of detectable classes.")
    class_names: list[str] = Field(default_factory=list, description="Model class names.")
    note: str = Field(..., description="Plain-language note about model state.")
