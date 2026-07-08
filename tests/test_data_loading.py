#!/usr/bin/env python3

"""test_data_loading.py -- tests the data-preparation and validation scripts.

Uses small synthetic images written to temporary directories to confirm the
raw-data quality checks, the annotation-readiness preparation steps, and the
YOLO label-format validator behave correctly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

import generate_annotation_guide as gag
import google_drive_sync as gds
import prepare_dataset as pds
import validate_data as vd


def _write_image(path: Path, width: int, height: int, value: int = 120) -> None:
    """Write a solid-color image to a path.

    Args:
        path: Destination image path.
        width: Image width in pixels.
        height: Image height in pixels.
        value: Fill intensity.
    """
    cv2.imwrite(str(path), np.full((height, width, 3), value, np.uint8))


def test_integrity_check_flags_corrupt_image(tmp_path: Path) -> None:
    """The integrity check flags a file that cannot be decoded."""
    good = tmp_path / "good.jpg"
    _write_image(good, 400, 400)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    failures = vd.check_image_integrity([good, bad])
    assert failures == [bad]


def test_duplicate_check_groups_identical_files(tmp_path: Path) -> None:
    """The duplicate check groups byte-identical images together."""
    original = tmp_path / "a.jpg"
    _write_image(original, 400, 400)
    copy = tmp_path / "b.jpg"
    shutil.copy(original, copy)
    groups = vd.check_duplicate_images([original, copy])
    assert len(groups) == 1
    assert sorted(p.name for group in groups.values() for p in group) == ["a.jpg", "b.jpg"]


def test_size_check_flags_undersized_image(tmp_path: Path) -> None:
    """The size check flags images below the minimum dimensions."""
    small = tmp_path / "small.png"
    _write_image(small, 200, 200)
    big = tmp_path / "big.png"
    _write_image(big, 400, 400)
    undersized = vd.check_image_sizes([small, big], (320, 320))
    assert undersized == [small]


def test_prepare_copies_images_and_creates_empty_labels(tmp_path: Path) -> None:
    """Preparation copies images and creates one empty label per image."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(3):
        _write_image(raw / f"s{i}.jpg", 400, 400, value=i * 30 + 10)
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    copied = pds.organize_raw_images(raw, images_dir)
    pds.create_placeholder_labels(images_dir, labels_dir)

    assert len(copied) == 3
    assert len(list(images_dir.glob("*.jpg"))) == 3
    labels = sorted(labels_dir.glob("*.txt"))
    assert len(labels) == 3
    assert all(label.stat().st_size == 0 for label in labels)


def test_manifest_records_each_image(tmp_path: Path) -> None:
    """The image manifest records one full record per image."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_image(images_dir / "s0.jpg", 320, 240)
    manifest = tmp_path / "manifest.json"
    pds.generate_image_manifest(images_dir, manifest)
    import json

    records = json.loads(manifest.read_text())
    assert len(records) == 1
    assert {"filename", "width", "height", "size_bytes", "sha256"} <= records[0].keys()
    assert records[0]["width"] == 320 and records[0]["height"] == 240


def test_annotation_format_validator_accepts_valid_and_empty(tmp_path: Path) -> None:
    """The label validator accepts a valid line and an empty file."""
    valid = tmp_path / "valid.txt"
    valid.write_text("0 0.5 0.5 0.2 0.2\n")
    assert gag.validate_annotation_format(valid) == []
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert gag.validate_annotation_format(empty) == []


def test_annotation_format_validator_reports_errors(tmp_path: Path) -> None:
    """The label validator reports wrong class ids and out-of-range coordinates."""
    bad = tmp_path / "bad.txt"
    bad.write_text("1 0.5 0.5 0.2 0.2\n0 1.5 0.5 0.2 0.2\n")
    errors = gag.validate_annotation_format(bad)
    assert len(errors) == 2


def test_drive_sync_accepts_plain_filename(tmp_path: Path) -> None:
    """A normal Drive filename resolves to a path inside the destination."""
    target = gds._safe_download_target(tmp_path, "specimen_001.jpg")
    assert target == tmp_path / "specimen_001.jpg"


def test_drive_sync_never_escapes_destination(tmp_path: Path) -> None:
    """Hostile Drive filenames are refused or confined inside the destination.

    A traversal or absolute name must never resolve to a path outside the
    download directory; it is either refused (``None``) or reduced to a bare
    filename that lands directly inside it.
    """
    base = tmp_path.resolve()
    for hostile in ["../../etc/cron.d/evil", "/etc/passwd", "sub/../../escape"]:
        target = gds._safe_download_target(tmp_path, hostile)
        assert target is not None
        assert target.resolve().parent == base

    for refused in ["", ".", ".."]:
        assert gds._safe_download_target(tmp_path, refused) is None
