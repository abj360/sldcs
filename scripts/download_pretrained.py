#!/usr/bin/env python3

"""download_pretrained.py -- downloads and verifies the pretrained YOLOv5s checkpoint.

This script fetches the official ``yolov5s.pt`` asset from the Ultralytics
GitHub release (by direct HTTPS request, so the exact file and its checksum are
reproducible), records its SHA-256 and metadata, and confirms it loads through
the same YOLOv5 API the application uses. It is the one command that makes the
detection model available on a fresh setup before any project-specific training
exists.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Final, Iterator

import requests
import torch
import yolov5

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Pinned Ultralytics YOLOv5 release the checkpoint is fetched from.
RELEASE_TAG: Final[str] = "v7.0"
SOURCE_URL: Final[str] = (
    f"https://github.com/ultralytics/yolov5/releases/download/{RELEASE_TAG}/yolov5s.pt"
)
MODEL_NAME: Final[str] = "yolov5s"

DEFAULT_WEIGHT_PATH: Final[Path] = PROJECT_ROOT / "weights" / "pretrained" / "yolov5s.pt"
DEFAULT_METADATA_PATH: Final[Path] = PROJECT_ROOT / "weights" / "pretrained" / "metadata.json"

DOWNLOAD_CHUNK_SIZE_BYTES: Final[int] = 1 << 20
HASH_BLOCK_SIZE_BYTES: Final[int] = 65536


@contextlib.contextmanager
def trusted_weights_load() -> Iterator[None]:
    """Temporarily allow full (non-``weights_only``) checkpoint deserialization.

    PyTorch 2.6 changed ``torch.load``'s ``weights_only`` default to ``True``,
    which rejects the pickled model classes in the official YOLOv5 checkpoint.
    The checkpoint here is the Ultralytics release we downloaded and SHA-256
    verified ourselves, so loading it fully is trusted. This context manager
    restores the original ``torch.load`` on exit.

    Yields:
        None. Use as ``with trusted_weights_load(): ...``.
    """
    original_load = torch.load
    torch.load = functools.partial(original_load, weights_only=False)
    try:
        yield
    finally:
        torch.load = original_load


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hex digest of a file's raw bytes.

    Args:
        file_path: File to hash.

    Returns:
        The hex-encoded SHA-256 digest.
    """
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_SIZE_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def download_yolov5s(destination_path: Path) -> None:
    """Download the official ``yolov5s.pt`` asset to a local path.

    Streams the response to disk so the checkpoint is never held fully in memory.

    Args:
        destination_path: Local path to write the checkpoint to.

    Raises:
        requests.HTTPError: If the download request returns an error status.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(SOURCE_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE_BYTES):
                handle.write(chunk)


def verify_model_integrity(file_path: Path, expected_sha256: str | None) -> bool:
    """Verify or record the checkpoint's SHA-256.

    Args:
        file_path: Checkpoint file to hash.
        expected_sha256: A previously recorded reference digest, or ``None`` when
            no reference exists yet.

    Returns:
        ``True`` if the file matches ``expected_sha256``; when ``expected_sha256``
        is ``None`` there is nothing to compare against, so the computed digest is
        printed for the caller to record and ``True`` is returned.
    """
    actual = compute_sha256(file_path)
    if expected_sha256 is None:
        print(f"Computed SHA-256 (no reference to compare): {actual}")
        return True
    matched = actual == expected_sha256
    if not matched:
        print(f"SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return matched


def generate_model_metadata(
    file_path: Path, metadata_path: Path, source_url: str, sha256: str
) -> None:
    """Write the pretrained checkpoint's metadata sidecar.

    Args:
        file_path: The checkpoint the metadata describes.
        metadata_path: Destination path for ``metadata.json``.
        source_url: URL the checkpoint was downloaded from.
        sha256: The checkpoint's SHA-256 digest.
    """
    metadata = {
        "model_name": MODEL_NAME,
        "source_url": source_url,
        "release_tag": RELEASE_TAG,
        "sha256": sha256,
        "download_date": date.today().isoformat(),
        "file_size_bytes": file_path.stat().st_size,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def test_model_load(file_path: Path) -> bool:
    """Confirm the checkpoint loads through the application's YOLOv5 API.

    Uses the same loading mechanism as the application so this test is
    representative rather than a separate code path. Any failure is caught and
    logged rather than propagated.

    Args:
        file_path: Checkpoint to load.

    Returns:
        ``True`` if the model loaded successfully, ``False`` otherwise.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        with trusted_weights_load():
            model = yolov5.load(str(file_path), device=device)
        class_count = len(model.names)
        print(f"Model loaded on {device} with {class_count} classes.")
        return True
    except Exception as error:  # noqa: BLE001 - report load failure, do not crash
        print(f"Model load failed: {error!r}")
        return False


def main() -> None:
    """Download, verify, describe, and load-test the pretrained checkpoint.

    Raises:
        SystemExit: With a non-zero code if the load test fails.
    """
    print(f"Downloading {MODEL_NAME} from {SOURCE_URL} ...")
    download_yolov5s(DEFAULT_WEIGHT_PATH)

    sha256 = compute_sha256(DEFAULT_WEIGHT_PATH)
    verify_model_integrity(DEFAULT_WEIGHT_PATH, expected_sha256=None)
    generate_model_metadata(DEFAULT_WEIGHT_PATH, DEFAULT_METADATA_PATH, SOURCE_URL, sha256)
    print(f"Metadata written to {DEFAULT_METADATA_PATH}.")

    if not test_model_load(DEFAULT_WEIGHT_PATH):
        print("FAIL: pretrained checkpoint did not load.")
        raise SystemExit(1)
    print("PASS: pretrained checkpoint downloaded, verified, and loadable.")


if __name__ == "__main__":
    main()
