"""Generate the shrimp larvae annotation guide and its example images.

This script assembles ``docs/ANNOTATION_GUIDE.md`` — the single reference a human
annotator uses to label the specimen images — together with three synthetic
example diagrams illustrating correct and incorrect bounding boxes. It also
exposes a validator for the YOLO label format so annotators can check their own
files. The example diagrams are placeholders drawn from simple shapes, not real
specimen photographs; the guide states this explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

# Select the non-interactive backend before importing pyplot so the script runs
# headlessly (e.g. over SSH or in CI) with no display attached.
matplotlib.use("Agg")

import matplotlib.patches as patches  # noqa: E402 - must follow backend selection
import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GUIDE_PATH: Final[Path] = PROJECT_ROOT / "docs" / "ANNOTATION_GUIDE.md"
EXAMPLES_DIR: Final[Path] = PROJECT_ROOT / "docs" / "annotation_examples"

# SLDCS visual-identity tokens reused so the guide's diagrams match the product.
COLOR_ABYSS: Final[str] = "#0B1620"
COLOR_TEAL: Final[str] = "#2BE1CB"
COLOR_CORAL: Final[str] = "#FF6E52"
COLOR_TEXT: Final[str] = "#EAF4F2"

ALLOWED_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
LARVAE_CLASS_ID: Final[int] = 0


def create_annotation_instructions() -> str:
    """Return the Markdown body describing how to annotate a larva.

    Covers the visual appearance of postlarval shrimp, the exact YOLO label line
    format with a worked numeric example, the tight-box rule, and the rule that
    touching larvae are annotated separately.

    Returns:
        The instructions as a Markdown string.
    """
    return (
        "## What a shrimp larva looks like\n\n"
        "At the postlarval stage relevant here, a shrimp larva is small "
        "(roughly a few millimetres long), translucent to faintly pigmented, and "
        "has a distinct curved, elongated body tapering from a rounded head "
        "region to a thin tail. In a tray image it reads as a small comma- or "
        "crescent-shaped translucent object against the darker water.\n\n"
        "## The YOLO label format\n\n"
        "Each image has a matching `.txt` label file with the same base name. "
        "Every line describes one larva, with five whitespace-separated fields:\n\n"
        "```\n<class_id> <x_center> <y_center> <width> <height>\n```\n\n"
        "- `class_id` is always `0` (the single class, `larvae`).\n"
        "- `x_center`, `y_center`, `width`, `height` are all normalized to the "
        "image dimensions, so every value is a float between `0` and `1`.\n\n"
        "**Worked example.** A larva whose tight box spans pixels x = 320..480 "
        "and y = 180..300 in a 1600x1200 image has:\n\n"
        "- center x = (320 + 480) / 2 / 1600 = `0.250000`\n"
        "- center y = (180 + 300) / 2 / 1200 = `0.200000`\n"
        "- width = (480 - 320) / 1600 = `0.100000`\n"
        "- height = (300 - 180) / 1200 = `0.100000`\n\n"
        "giving the label line:\n\n"
        "```\n0 0.250000 0.200000 0.100000 0.100000\n```\n\n"
        "An image with no larvae has an **empty** label file — that is valid and "
        "means \"no objects\".\n\n"
        "## Box tightness\n\n"
        "Draw the box tight around the entire visible body of the larva, "
        "including the tail, with no more than a small fixed pixel margin. Do not "
        "leave large empty borders, and do not clip any visible part of the "
        "animal.\n\n"
        "## Touching and overlapping larvae\n\n"
        "When two or more larvae touch or overlap, annotate each one as its own "
        "separate box. Never merge several larvae into a single box, even when "
        "their bodies are in contact.\n"
    )


def _style_axes(ax: plt.Axes, title: str) -> None:
    """Apply the shared dark-figure styling to an example axis.

    Args:
        ax: The axis to style.
        title: Caption drawn beneath the diagram.
    """
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_facecolor(COLOR_ABYSS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=COLOR_TEXT, fontsize=11)


def _draw_larva(ax: plt.Axes, cx: float, cy: float, rx: float, ry: float) -> None:
    """Draw a single stylized larva as a filled ellipse.

    Args:
        ax: Target axis.
        cx: Ellipse center x.
        cy: Ellipse center y.
        rx: Half-width.
        ry: Half-height.
    """
    ax.add_patch(
        patches.Ellipse(
            (cx, cy), rx * 2, ry * 2, facecolor=COLOR_TEAL, edgecolor="none", alpha=0.55
        )
    )


def create_sample_annotations(source_images: list[Path], output_dir: Path) -> list[Path]:
    """Generate three placeholder example diagrams for the guide.

    Because no real annotated specimen photograph exists yet, this draws simple
    synthetic shapes rather than using real images. The three diagrams show a
    correctly tight box, a box drawn too loose with the excess shaded coral, and
    two adjacent larvae wrongly merged into one box with a dashed line marking the
    correct split. Filenames carry a ``PLACEHOLDER`` marker.

    Args:
        source_images: Real images to reference if any exist; unused for drawing
            but accepted so the signature is stable once real examples replace
            these placeholders.
        output_dir: Directory to write the PNG diagrams into.

    Returns:
        The paths of the three generated PNG files, in example order.
    """
    del source_images  # Placeholders are drawn synthetically; see docstring.
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Example 1 — correctly tight box.
    fig, ax = plt.subplots(figsize=(4, 4))
    _style_axes(ax, "1. Correct: tight box around the whole larva")
    _draw_larva(ax, 50, 50, 22, 10)
    ax.add_patch(
        patches.Rectangle((26, 38), 48, 24, fill=False, edgecolor=COLOR_TEAL, linewidth=2)
    )
    path1 = output_dir / "example_1_tight_PLACEHOLDER.png"
    fig.savefig(path1, facecolor=COLOR_ABYSS, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(path1)

    # Example 2 — box too loose, excess shaded coral.
    fig, ax = plt.subplots(figsize=(4, 4))
    _style_axes(ax, "2. Wrong: box too loose (excess in coral)")
    ax.add_patch(
        patches.Rectangle((12, 20), 76, 60, facecolor=COLOR_CORAL, edgecolor="none", alpha=0.28)
    )
    _draw_larva(ax, 50, 50, 22, 10)
    ax.add_patch(
        patches.Rectangle((12, 20), 76, 60, fill=False, edgecolor=COLOR_TEAL, linewidth=2)
    )
    path2 = output_dir / "example_2_loose_PLACEHOLDER.png"
    fig.savefig(path2, facecolor=COLOR_ABYSS, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(path2)

    # Example 3 — two larvae wrongly merged into one box; dashed line = correct split.
    fig, ax = plt.subplots(figsize=(4, 4))
    _style_axes(ax, "3. Wrong: two larvae merged (dashed = correct split)")
    _draw_larva(ax, 38, 50, 16, 9)
    _draw_larva(ax, 64, 50, 16, 9)
    ax.add_patch(
        patches.Rectangle((20, 38), 62, 24, fill=False, edgecolor=COLOR_TEAL, linewidth=2)
    )
    ax.plot([51, 51], [36, 64], color=COLOR_CORAL, linestyle="--", linewidth=1.5)
    path3 = output_dir / "example_3_merged_PLACEHOLDER.png"
    fig.savefig(path3, facecolor=COLOR_ABYSS, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(path3)

    return written


def validate_annotation_format(label_path: Path) -> list[str]:
    """Validate one YOLO label file's syntax.

    Args:
        label_path: Path to the ``.txt`` label file.

    Returns:
        A list of human-readable error strings, one per malformed line; empty if
        the file is fully valid (an empty file is valid).
    """
    errors: list[str] = []
    text = label_path.read_text(encoding="utf-8")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(
                f"Line {line_number}: expected 5 fields, found {len(fields)}: '{line}'"
            )
            continue
        class_field, *coord_fields = fields
        if class_field != str(LARVAE_CLASS_ID):
            errors.append(
                f"Line {line_number}: class id must be {LARVAE_CLASS_ID}, found '{class_field}'"
            )
        for name, value in zip(("x_center", "y_center", "width", "height"), coord_fields):
            try:
                number = float(value)
            except ValueError:
                errors.append(f"Line {line_number}: {name} is not a number: '{value}'")
                continue
            if not 0.0 <= number <= 1.0:
                errors.append(
                    f"Line {line_number}: {name} must be within [0, 1], found {number}"
                )
    return errors


def create_qa_checklist() -> str:
    """Return the annotator QA checklist as Markdown.

    Returns:
        A Markdown checklist, one item per quality criterion.
    """
    items = [
        "Every box is tight around the whole visible larva.",
        "No visible larva is left un-annotated.",
        "No box is drawn where there is no larva (no false positives).",
        "Similarly sized larvae have consistently sized boxes.",
        "Every line uses class id `0` and only class `0`.",
        "Each label file's name matches its image exactly (same base name).",
        "Every line has exactly five whitespace-separated fields.",
    ]
    return "## Quality-assurance checklist\n\n" + "\n".join(f"- [ ] {item}" for item in items) + "\n"


def _list_images(directory: Path) -> list[Path]:
    """List image files in a directory by allowed extension.

    Args:
        directory: Directory to scan; may not exist.

    Returns:
        Sorted image paths, or an empty list if the directory is absent.
    """
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )


def main() -> None:
    """Assemble the annotation guide and write it with embedded example images."""
    source_images = _list_images(PROJECT_ROOT / "data" / "processed" / "images")
    example_paths = create_sample_annotations(source_images, EXAMPLES_DIR)

    relative_examples = [path.relative_to(GUIDE_PATH.parent) for path in example_paths]
    captions = [
        "Correctly tight box.",
        "Box drawn too loose; excess area shaded coral.",
        "Two adjacent larvae wrongly merged; dashed line shows the correct split.",
    ]
    image_block = "\n\n".join(
        f"![{caption}]({rel.as_posix()})\n\n*{caption}*"
        for rel, caption in zip(relative_examples, captions)
    )

    document = "\n".join(
        [
            "# SLDCS annotation guide",
            "",
            "> **Note:** The three example images below are synthetic placeholders "
            "drawn from simple shapes, not real specimen photographs. Replace them "
            "with real annotated examples once real annotated images exist.",
            "",
            create_annotation_instructions(),
            "## Example annotations",
            "",
            image_block,
            "",
            create_qa_checklist(),
        ]
    )
    GUIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GUIDE_PATH.write_text(document, encoding="utf-8")
    print(f"Annotation guide written to {GUIDE_PATH}.")
    for path in example_paths:
        print(f"  example: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
