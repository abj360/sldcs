#!/usr/bin/env python3

"""conftest.py -- provides shared pytest fixtures for the test suite.

Provides a stubbed inference engine so the API and pipeline can be tested
without loading real model weights or a GPU, a test client wired to that stub,
and a path to a small sample image fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class StubInference:
    """A drop-in replacement for :class:`ModelInference` used in tests.

    Returns deterministic detections without loading any weights, so API and
    route tests are fast and require no GPU. It mirrors the public surface the
    application depends on and nothing more.

    Attributes:
        device: The reported compute device string.
    """

    def __init__(self) -> None:
        """Initialize the stub with a fixed device label."""
        self.device = "cpu"

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        """Return two fixed detections regardless of image content.

        Args:
            image: The input image (used only for its dimensions).

        Returns:
            A detection outcome dict shaped like the real engine's.
        """
        height, width = image.shape[:2]
        detections = [
            {
                "x1": 0.1 * width, "y1": 0.1 * height, "x2": 0.2 * width, "y2": 0.2 * height,
                "confidence": 0.91, "class_id": 0, "class_name": "larvae",
            },
            {
                "x1": 0.5 * width, "y1": 0.5 * height, "x2": 0.6 * width, "y2": 0.6 * height,
                "confidence": 0.44, "class_id": 0, "class_name": "larvae",
            },
        ]
        return {"detections": detections, "tiles_scanned": 3, "duplicates_merged": 1}

    def detect_batch(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        """Run :meth:`detect` on each image.

        Args:
            images: Input images.

        Returns:
            One detection outcome per image.
        """
        return [self.detect(image) for image in images]

    def get_available_classes(self) -> list[str]:
        """Return the stub's single class name.

        Returns:
            ``["larvae"]``.
        """
        return ["larvae"]

    def get_metadata(self) -> dict[str, Any]:
        """Return stub model metadata.

        Returns:
            Metadata mirroring the real engine's shape.
        """
        return {
            "model_path": "stub",
            "device": self.device,
            "class_count": 1,
            "class_names": ["larvae"],
        }


@pytest.fixture
def stub_inference() -> StubInference:
    """Provide a fresh stub inference engine.

    Returns:
        A new :class:`StubInference` instance.
    """
    return StubInference()


@pytest.fixture
def test_client(stub_inference: StubInference) -> Iterator[TestClient]:
    """Provide a TestClient backed by the stub engine.

    Injects the stub and real settings into the app state without running the
    startup event (which would load real weights), then clears them afterward.

    Args:
        stub_inference: The stub engine to serve detections with.

    Yields:
        A configured :class:`fastapi.testclient.TestClient`.
    """
    app.state.inference = stub_inference
    app.state.settings = get_settings()
    client = TestClient(app)
    yield client
    app.state.inference = None


@pytest.fixture
def sample_image_path() -> Path:
    """Provide the path to the sample image fixture.

    Returns:
        Path to ``tests/fixtures/sample_image.jpg``.
    """
    return FIXTURES_DIR / "sample_image.jpg"


@pytest.fixture
def sample_image_bytes(sample_image_path: Path) -> bytes:
    """Provide the raw bytes of the sample image fixture.

    Args:
        sample_image_path: Path to the fixture image.

    Returns:
        The fixture image's bytes.
    """
    return sample_image_path.read_bytes()
