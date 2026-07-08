"""Quality-check the raw specimen images and report problems.

This script inspects the files in ``data/raw`` for three classes of defect that
would silently corrupt training or inference: unreadable/corrupted images,
byte-for-byte duplicates, and images smaller than the configured minimum size.
It reads every threshold from ``config/data_config.yaml`` and writes a Markdown
report; it never modifies or deletes any image.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Final

import cv2
import yaml

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "config" / "data_config.yaml"
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
REPORT_PATH: Final[Path] = RAW_DIR / "validation_report.md"

# Read in 64 KB blocks when hashing so arbitrarily large images never have to be
# held in memory all at once.
HASH_BLOCK_SIZE_BYTES: Final[int] = 65536


def load_data_config(config_path: Path = DATA_CONFIG_PATH) -> dict:
    """Load the raw-data quality-control configuration.

    Args:
        config_path: Path to ``data_config.yaml``.

    Returns:
        The parsed configuration mapping.
    """
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def collect_image_paths(directory: Path, allowed_formats: list[str]) -> list[Path]:
    """List image files in a directory whose extension is allowed.

    Args:
        directory: Directory to scan (non-recursively).
        allowed_formats: Allowed extensions without leading dots, e.g. ``["jpg"]``.

    Returns:
        Sorted list of matching image paths.
    """
    allowed = {f".{fmt.lower().lstrip('.')}" for fmt in allowed_formats}
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    )


def _hash_file(path: Path) -> str:
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


def check_image_integrity(image_paths: list[Path]) -> list[Path]:
    """Find images that cannot be decoded.

    OpenCV returns ``None`` rather than raising on a corrupted or truncated
    image, so an explicit ``None`` check is required instead of a try/except.

    Args:
        image_paths: Images to test.

    Returns:
        The subset of paths that failed to decode.
    """
    failures: list[Path] = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            failures.append(path)
    return failures


def check_duplicate_images(image_paths: list[Path]) -> dict[str, list[Path]]:
    """Group images that are byte-for-byte identical.

    A SHA-256 content hash is used rather than a perceptual hash because these
    are direct camera exports, not edited copies, so exact-byte matching is both
    stricter and simpler to reason about.

    Args:
        image_paths: Images to compare.

    Returns:
        A mapping of content hash to the paths sharing it, containing only hashes
        with more than one file.
    """
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        by_hash[_hash_file(path)].append(path)
    return {digest: paths for digest, paths in by_hash.items() if len(paths) > 1}


def check_image_sizes(image_paths: list[Path], min_size: tuple[int, int]) -> list[Path]:
    """Find images smaller than the minimum allowed dimensions.

    Args:
        image_paths: Images to measure.
        min_size: Minimum allowed ``(width, height)`` in pixels.

    Returns:
        Paths whose width or height is below the corresponding minimum. Images
        that cannot be read are skipped here; they are reported by
        :func:`check_image_integrity`.
    """
    min_width, min_height = min_size
    undersized: list[Path] = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        if width < min_width or height < min_height:
            undersized.append(path)
    return undersized


def _format_section(title: str, paths: list[Path], base: Path) -> str:
    """Render one report section listing offending files or 'None found'.

    Args:
        title: Section heading.
        paths: Offending paths for this check.
        base: Directory to render paths relative to.

    Returns:
        The Markdown section text.
    """
    lines = [f"## {title}", ""]
    if not paths:
        lines.append("None found.")
    else:
        for path in paths:
            lines.append(f"- `{path.relative_to(base)}`")
    lines.append("")
    return "\n".join(lines)


def generate_data_report(
    integrity_failures: list[Path],
    duplicate_groups: dict[str, list[Path]],
    undersized: list[Path],
    report_path: Path,
) -> None:
    """Write a Markdown report with one section per quality check.

    Args:
        integrity_failures: Paths that failed to decode.
        duplicate_groups: Duplicate groups keyed by content hash.
        undersized: Paths below the minimum size.
        report_path: Destination for the Markdown report.
    """
    base = report_path.parent
    duplicate_paths = [path for group in duplicate_groups.values() for path in group]
    body = [
        "# Raw dataset validation report",
        "",
        _format_section("Corrupted / unreadable images", integrity_failures, base),
        _format_section("Duplicate images", duplicate_paths, base),
        _format_section("Undersized images", undersized, base),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    """Run all quality checks against ``data/raw`` and write the report."""
    config = load_data_config()
    image_paths = collect_image_paths(RAW_DIR, config["ALLOWED_FORMATS"])
    min_size = tuple(config["MIN_IMAGE_SIZE"])

    integrity_failures = check_image_integrity(image_paths)
    duplicate_groups = check_duplicate_images(image_paths)
    undersized = check_image_sizes(image_paths, min_size)

    generate_data_report(integrity_failures, duplicate_groups, undersized, REPORT_PATH)

    print(f"Validated {len(image_paths)} image(s) in {RAW_DIR}.")
    print(f"  Corrupted:  {len(integrity_failures)}")
    print(f"  Duplicates: {sum(len(g) for g in duplicate_groups.values())}")
    print(f"  Undersized: {len(undersized)}")
    print(f"Report written to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
