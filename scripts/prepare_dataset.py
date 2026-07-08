"""Prepare the raw specimen images for annotation.

This script takes the validated raw images and produces the working copy the
annotation tool operates on: a flat directory of images, a matching set of empty
YOLO label files, a manifest describing every image, a human-readable summary,
and an off-tree backup of the immutable raw data. It copies rather than moves, so
``data/raw`` remains the untouched source of truth. It does not split the data
into train/val/test — that happens after annotation, in
``scripts/prepare_train_data.py``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Final

from PIL import Image

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_IMAGES_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "images"
PROCESSED_LABELS_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "labels"
MANIFEST_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "image_manifest.json"
SUMMARY_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "dataset_summary.md"

# The backup lives outside the project tree so it is never swept into a git
# operation or a Docker build context.
BACKUP_ROOT: Final[Path] = PROJECT_ROOT.parent / "sldcs_raw_backups"

ALLOWED_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
HASH_BLOCK_SIZE_BYTES: Final[int] = 65536


def _file_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file's raw bytes.

    Args:
        path: File to hash.

    Returns:
        The hex-encoded SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_SIZE_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _list_images(directory: Path) -> list[Path]:
    """List image files in a directory by allowed extension.

    Args:
        directory: Directory to scan (non-recursively).

    Returns:
        Sorted list of image paths with an allowed suffix.
    """
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )


def organize_raw_images(raw_dir: Path, processed_images_dir: Path) -> list[Path]:
    """Copy every valid raw image into a flat processed directory.

    Args:
        raw_dir: Immutable source directory of raw images.
        processed_images_dir: Destination directory for the working copy.

    Returns:
        The destination paths of every copied image.
    """
    processed_images_dir.mkdir(parents=True, exist_ok=True)
    destinations: list[Path] = []
    for source in _list_images(raw_dir):
        destination = processed_images_dir / source.name
        shutil.copy2(source, destination)
        destinations.append(destination)
    return destinations


def generate_image_manifest(images_dir: Path, manifest_path: Path) -> None:
    """Write a JSON manifest describing every image in a directory.

    Each record contains the filename, pixel width and height, file size in
    bytes, and SHA-256 content hash.

    Args:
        images_dir: Directory of images to describe.
        manifest_path: Destination path for the JSON manifest.
    """
    records: list[dict] = []
    for path in _list_images(images_dir):
        with Image.open(path) as image:
            width, height = image.size
        records.append(
            {
                "filename": path.name,
                "width": width,
                "height": height,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def create_placeholder_labels(images_dir: Path, labels_dir: Path) -> None:
    """Create one empty YOLO label file per image.

    An empty label file is valid YOLO syntax meaning "no objects"; the
    annotation tool overwrites it. Existing label files are left untouched so
    annotation work is never clobbered by re-running preparation.

    Args:
        images_dir: Directory of images needing labels.
        labels_dir: Destination directory for the ``.txt`` label files.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    for path in _list_images(images_dir):
        label_path = labels_dir / f"{path.stem}.txt"
        if not label_path.exists():
            label_path.touch()


def backup_raw_data(raw_dir: Path, backup_dir: Path) -> Path:
    """Copy the raw directory tree into a timestamped backup outside the tree.

    Args:
        raw_dir: Directory to back up.
        backup_dir: Root under which a timestamped subdirectory is created.

    Returns:
        The path of the timestamped backup directory that was created.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = backup_dir / f"raw_{stamp}"
    shutil.copytree(raw_dir, destination)
    return destination


def generate_dataset_summary(manifest_path: Path, summary_path: Path) -> None:
    """Write a Markdown summary of the dataset from its manifest.

    Reports the total image count, average width and height, minimum and maximum
    dimensions, and a breakdown of file extensions present.

    Args:
        manifest_path: Path to the JSON manifest.
        summary_path: Destination for the Markdown summary.
    """
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = ["# Dataset summary", ""]
    if not records:
        lines.append("No images found.")
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    widths = [record["width"] for record in records]
    heights = [record["height"] for record in records]
    extensions = Counter(Path(record["filename"]).suffix.lower() for record in records)

    lines.extend(
        [
            f"- Total images: {len(records)}",
            f"- Average dimensions (w x h): {sum(widths) / len(widths):.1f} x {sum(heights) / len(heights):.1f}",
            f"- Minimum dimensions (w x h): {min(widths)} x {min(heights)}",
            f"- Maximum dimensions (w x h): {max(widths)} x {max(heights)}",
            "",
            "## File extensions",
            "",
        ]
    )
    for suffix, count in sorted(extensions.items()):
        lines.append(f"- `{suffix}`: {count}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full dataset-preparation pipeline in order."""
    copied = organize_raw_images(RAW_DIR, PROCESSED_IMAGES_DIR)
    generate_image_manifest(PROCESSED_IMAGES_DIR, MANIFEST_PATH)
    create_placeholder_labels(PROCESSED_IMAGES_DIR, PROCESSED_LABELS_DIR)
    backup_path = backup_raw_data(RAW_DIR, BACKUP_ROOT)
    generate_dataset_summary(MANIFEST_PATH, SUMMARY_PATH)

    print(f"Prepared {len(copied)} image(s) for annotation.")
    print(f"  Images:   {PROCESSED_IMAGES_DIR}")
    print(f"  Labels:   {PROCESSED_LABELS_DIR}")
    print(f"  Manifest: {MANIFEST_PATH}")
    print(f"  Summary:  {SUMMARY_PATH}")
    print(f"  Backup:   {backup_path}")


if __name__ == "__main__":
    main()
