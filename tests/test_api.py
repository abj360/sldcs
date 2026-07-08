#!/usr/bin/env python3

"""test_api.py -- tests the SLDCS API endpoints.

These tests exercise every endpoint through the test client using the stubbed
inference engine, so they validate request handling, validation, and response
shapes without loading real weights.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_reports_model_loaded(test_client: TestClient) -> None:
    """Health endpoint reports the model is loaded when an engine is present."""
    response = test_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_ready_returns_200_when_loaded(test_client: TestClient) -> None:
    """Readiness endpoint returns 200 with ready=true when the engine is loaded."""
    response = test_client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}


def test_version_returns_app_version(test_client: TestClient) -> None:
    """Version endpoint returns the application version string."""
    response = test_client.get("/version")
    assert response.status_code == 200
    assert "version" in response.json()


def test_config_returns_pipeline_parameters(test_client: TestClient) -> None:
    """Config endpoint exposes the confidence threshold and tile parameters."""
    response = test_client.get("/config")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"conf_threshold", "tile_size", "tile_overlap"}


def test_model_info_is_honest_about_pretrained(test_client: TestClient) -> None:
    """Model info reports the pretrained checkpoint and no project training."""
    response = test_client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["trained_on_project_data"] is False
    assert body["version"] == "pretrained_yolov5s"


def test_model_classes_lists_names(test_client: TestClient) -> None:
    """Model classes endpoint returns the stub's class names."""
    response = test_client.get("/model/classes")
    assert response.status_code == 200
    assert response.json() == ["larvae"]


def test_detect_returns_result_with_annotation(
    test_client: TestClient, sample_image_bytes: bytes
) -> None:
    """Single detection returns a full result including an annotated image."""
    response = test_client.post(
        "/detect",
        files={"files": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    result = results[0]
    assert result["larvae_count"] == 2
    assert result["tiles_scanned"] == 3
    assert result["duplicates_merged"] == 1
    assert result["annotated_image"]
    assert len(result["confidence_distribution"]) == 10
    assert len(result["detections"]) == 2


def test_batch_detect_returns_counts_only(
    test_client: TestClient, sample_image_bytes: bytes
) -> None:
    """Batch detection aggregates counts and omits image encoding."""
    response = test_client.post(
        "/batch-detect",
        files=[
            ("files", ("a.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("b.jpg", sample_image_bytes, "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["image_count"] == 2
    assert body["total_larvae"] == 4
    assert body["average_per_image"] == 2.0
    assert all("annotated_image" not in item for item in body["results"])


def test_detect_rejects_unsupported_type(test_client: TestClient) -> None:
    """An unsupported file type is rejected with HTTP 415."""
    response = test_client.post(
        "/detect", files={"files": ("notes.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 415


def test_detect_rejects_empty_file(test_client: TestClient) -> None:
    """An empty upload is rejected with HTTP 400."""
    response = test_client.post(
        "/detect", files={"files": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400


def test_detect_rejects_oversized_file(
    test_client: TestClient, sample_image_bytes: bytes
) -> None:
    """An upload exceeding the configured size limit is rejected with HTTP 413."""
    original = test_client.app.state.settings.MAX_FILE_SIZE
    test_client.app.state.settings.MAX_FILE_SIZE = 10
    try:
        response = test_client.post(
            "/detect", files={"files": ("big.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 413
    finally:
        test_client.app.state.settings.MAX_FILE_SIZE = original


def test_detect_rejects_too_many_files(
    test_client: TestClient, sample_image_bytes: bytes
) -> None:
    """A request with more files than the per-request cap is rejected with 413."""
    from app.routes import MAX_UPLOAD_FILES

    files = [
        ("files", (f"img_{i}.jpg", sample_image_bytes, "image/jpeg"))
        for i in range(MAX_UPLOAD_FILES + 1)
    ]
    response = test_client.post("/detect", files=files)
    assert response.status_code == 413


def test_detect_rejects_decompression_bomb(
    test_client: TestClient, sample_image_bytes: bytes, monkeypatch
) -> None:
    """An image whose decoded pixel count exceeds the guard is rejected with 413."""
    monkeypatch.setattr("app.routes.MAX_IMAGE_PIXELS", 1)
    response = test_client.post(
        "/detect", files={"files": ("bomb.png", sample_image_bytes, "image/png")}
    )
    assert response.status_code == 413
