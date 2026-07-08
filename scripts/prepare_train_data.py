"""Split the annotated dataset into train/val/test partitions.

Once annotation is complete, this script partitions the flat image and label
sets into the train/val/test subdirectories the dataset descriptor points at,
copying files (leaving the flat originals intact) with a fixed random seed for
reproducibility, and verifies the split is disjoint and label-complete.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
IMAGES_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "images"
LABELS_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "labels"

ALLOWED_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "val", "test")
DEFAULT_RATIOS: Final[tuple[float, float, float]] = (0.8, 0.1, 0.1)
DEFAULT_SEED: Final[int] = 42


def _flat_images(images_dir: Path) -> list[Path]:
    """List images directly in a directory, ignoring split subdirectories.

    Args:
        images_dir: Directory to scan (top level only).

    Returns:
        Sorted image paths at the top level.
    """
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )


def split_dataset(
    images_dir: Path = IMAGES_DIR,
    labels_dir: Path = LABELS_DIR,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    """Partition flat images and labels into train/val/test subdirectories.

    Files are copied (not moved) into ``<images_dir>/<split>`` and
    ``<labels_dir>/<split>`` so the flat source set is preserved. Each image's
    matching ``.txt`` label is copied alongside it; images with no label file
    are skipped and not counted.

    Args:
        images_dir: Directory of flat images.
        labels_dir: Directory of flat label files.
        ratios: Train/val/test fractions; must sum to 1.
        seed: Random seed controlling the shuffle.

    Returns:
        A mapping of split name to the number of images placed in it.

    Raises:
        ValueError: If ``ratios`` does not sum to approximately 1.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios} (sum {sum(ratios)}).")

    paired = [
        image for image in _flat_images(images_dir) if (labels_dir / f"{image.stem}.txt").is_file()
    ]
    random.Random(seed).shuffle(paired)

    total = len(paired)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    partitions = {
        "train": paired[:train_end],
        "val": paired[train_end:val_end],
        "test": paired[val_end:],
    }

    counts: dict[str, int] = {}
    for split, items in partitions.items():
        image_out = images_dir / split
        label_out = labels_dir / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for image in items:
            shutil.copy2(image, image_out / image.name)
            shutil.copy2(labels_dir / f"{image.stem}.txt", label_out / f"{image.stem}.txt")
        counts[split] = len(items)
    return counts


def verify_split(images_dir: Path = IMAGES_DIR, labels_dir: Path = LABELS_DIR) -> dict:
    """Verify the split is disjoint and every split image has a label.

    Args:
        images_dir: Directory containing the split image subdirectories.
        labels_dir: Directory containing the split label subdirectories.

    Returns:
        A dict with ``ok`` (bool), ``counts`` (per split), ``overlaps`` (image
        stems appearing in more than one split), and ``missing_labels`` (split
        images lacking a label).
    """
    per_split_stems: dict[str, set[str]] = {}
    missing_labels: list[str] = []
    counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        image_split = images_dir / split
        label_split = labels_dir / split
        stems = {
            path.stem
            for path in (image_split.glob("*") if image_split.is_dir() else [])
            if path.suffix.lower() in ALLOWED_SUFFIXES
        }
        per_split_stems[split] = stems
        counts[split] = len(stems)
        for stem in stems:
            if not (label_split / f"{stem}.txt").is_file():
                missing_labels.append(f"{split}/{stem}")

    overlaps: list[str] = []
    splits = list(per_split_stems)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlaps.extend(sorted(per_split_stems[splits[i]] & per_split_stems[splits[j]]))

    return {
        "ok": not overlaps and not missing_labels,
        "counts": counts,
        "overlaps": overlaps,
        "missing_labels": missing_labels,
    }


def main() -> None:
    """Split the dataset from the command line and report the result."""
    parser = argparse.ArgumentParser(description="Split annotated data into train/val/test.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed.")
    args = parser.parse_args()

    counts = split_dataset(seed=args.seed)
    verification = verify_split()
    print(f"Split complete: {counts}")
    print(f"Verification: {'OK' if verification['ok'] else 'FAILED'} -> {verification}")
    if not verification["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
