"""Validate completed YOLO annotations against the images they label.

After a human has annotated the specimen images, this script checks that every
label file is syntactically valid YOLO, that its boxes lie within the image, and
that images and labels correspond one-to-one, then writes a report. It reads and
reports only; it never edits an annotation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
IMAGES_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "images"
LABELS_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "labels"
REPORT_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "annotation_report.md"

ALLOWED_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
LARVAE_CLASS_ID: Final[int] = 0
# Boxes narrower/shorter than this fraction of the image are flagged as suspiciously tiny.
MIN_BOX_FRACTION: Final[float] = 0.002


def validate_yolo_format(label_path: Path) -> list[str]:
    """Validate a label file's YOLO syntax.

    Args:
        label_path: Path to the ``.txt`` label file.

    Returns:
        Human-readable error strings for malformed lines; empty if valid (an
        empty file is valid).
    """
    errors: list[str] = []
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"Line {line_number}: expected 5 fields, found {len(fields)}.")
            continue
        if fields[0] != str(LARVAE_CLASS_ID):
            errors.append(f"Line {line_number}: class id must be {LARVAE_CLASS_ID}.")
        for name, value in zip(("x_center", "y_center", "width", "height"), fields[1:]):
            try:
                number = float(value)
            except ValueError:
                errors.append(f"Line {line_number}: {name} is not a number.")
                continue
            if not 0.0 <= number <= 1.0:
                errors.append(f"Line {line_number}: {name} out of [0, 1].")
    return errors


def validate_bbox_within_image(label_path: Path) -> list[str]:
    """Check that every normalized box lies fully within the image bounds.

    Args:
        label_path: Path to the ``.txt`` label file.

    Returns:
        Error strings for any box whose extent crosses an image edge; empty if
        all boxes are in bounds. Malformed lines are skipped (reported by
        :func:`validate_yolo_format`).
    """
    errors: list[str] = []
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if len(fields) != 5:
            continue
        try:
            _, x_center, y_center, width, height = (float(f) for f in fields)
        except ValueError:
            continue
        if x_center - width / 2 < 0 or x_center + width / 2 > 1:
            errors.append(f"Line {line_number}: box exceeds image width bounds.")
        if y_center - height / 2 < 0 or y_center + height / 2 > 1:
            errors.append(f"Line {line_number}: box exceeds image height bounds.")
    return errors


def match_images_to_labels(images_dir: Path, labels_dir: Path) -> dict:
    """Match images to their label files by base name.

    Args:
        images_dir: Directory of images.
        labels_dir: Directory of ``.txt`` label files.

    Returns:
        A dict with ``matched`` (int), ``images_without_labels`` (list of names),
        and ``labels_without_images`` (list of names).
    """
    image_stems = {
        path.stem
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    }
    label_stems = {path.stem for path in labels_dir.glob("*.txt")}
    return {
        "matched": len(image_stems & label_stems),
        "images_without_labels": sorted(image_stems - label_stems),
        "labels_without_images": sorted(label_stems - image_stems),
    }


def check_annotation_quality(label_path: Path) -> list[str]:
    """Flag suspicious (but syntactically valid) annotations.

    Args:
        label_path: Path to the ``.txt`` label file.

    Returns:
        Warning strings for zero-area or suspiciously tiny boxes; empty if none.
    """
    warnings: list[str] = []
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if len(fields) != 5:
            continue
        try:
            _, _, _, width, height = (float(f) for f in fields)
        except ValueError:
            continue
        if width <= 0 or height <= 0:
            warnings.append(f"Line {line_number}: zero-area box.")
        elif width < MIN_BOX_FRACTION or height < MIN_BOX_FRACTION:
            warnings.append(f"Line {line_number}: suspiciously tiny box.")
    return warnings


def generate_annotation_report(
    format_errors: dict[str, list[str]],
    bounds_errors: dict[str, list[str]],
    quality_warnings: dict[str, list[str]],
    match_summary: dict,
    report_path: Path,
) -> None:
    """Write the annotation-validation report as Markdown.

    Args:
        format_errors: Per-file format errors keyed by filename.
        bounds_errors: Per-file out-of-bounds errors keyed by filename.
        quality_warnings: Per-file quality warnings keyed by filename.
        match_summary: The output of :func:`match_images_to_labels`.
        report_path: Destination for the Markdown report.
    """

    def section(title: str, mapping: dict[str, list[str]]) -> str:
        lines = [f"## {title}", ""]
        offenders = {name: msgs for name, msgs in mapping.items() if msgs}
        if not offenders:
            lines.append("None found.")
        else:
            for name, messages in sorted(offenders.items()):
                lines.append(f"- `{name}`:")
                lines.extend(f"  - {message}" for message in messages)
        lines.append("")
        return "\n".join(lines)

    body = [
        "# Annotation validation report",
        "",
        "## Image/label matching",
        "",
        f"- Matched pairs: {match_summary['matched']}",
        f"- Images without labels: {len(match_summary['images_without_labels'])}",
        f"- Labels without images: {len(match_summary['labels_without_images'])}",
        "",
        section("Format errors", format_errors),
        section("Out-of-bounds boxes", bounds_errors),
        section("Quality warnings", quality_warnings),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    """Validate all annotations under ``data/processed`` and write the report."""
    match_summary = match_images_to_labels(IMAGES_DIR, LABELS_DIR)
    format_errors: dict[str, list[str]] = {}
    bounds_errors: dict[str, list[str]] = {}
    quality_warnings: dict[str, list[str]] = {}
    for label_path in sorted(LABELS_DIR.glob("*.txt")):
        format_errors[label_path.name] = validate_yolo_format(label_path)
        bounds_errors[label_path.name] = validate_bbox_within_image(label_path)
        quality_warnings[label_path.name] = check_annotation_quality(label_path)

    generate_annotation_report(
        format_errors, bounds_errors, quality_warnings, match_summary, REPORT_PATH
    )
    total_errors = sum(len(v) for v in format_errors.values()) + sum(
        len(v) for v in bounds_errors.values()
    )
    print(f"Validated {len(format_errors)} label file(s); {total_errors} error(s).")
    print(f"Report written to {REPORT_PATH}.")


if __name__ == "__main__":
    main()
