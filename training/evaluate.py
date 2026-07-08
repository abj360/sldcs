#!/usr/bin/env python3

"""evaluate.py -- evaluates trained models and renders evaluation figures.

Pure functions for measuring a trained YOLOv5 checkpoint against a dataset split
and for producing the evaluation figures (confusion matrix, precision-recall
curve, and annotated detection samples) in the system's own visual identity.
Nothing here trains a model or serves requests.
"""

from __future__ import annotations

import contextlib
import functools
from pathlib import Path
from typing import Any, Final, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yolov5.val as yolo_val  # noqa: E402

from app.utils import draw_boxes  # noqa: E402

# SLDCS identity colors for figures.
COLOR_ABYSS: Final[str] = "#0B1620"
COLOR_TEAL: Final[str] = "#2BE1CB"
COLOR_HAIRLINE: Final[str] = "#20404D"
COLOR_TEXT: Final[str] = "#EAF4F2"


@contextlib.contextmanager
def _trusted_weights_load() -> Iterator[None]:
    """Temporarily allow full deserialization of a trusted checkpoint.

    Yields:
        None.
    """
    original_load = torch.load
    torch.load = functools.partial(original_load, weights_only=False)
    try:
        yield
    finally:
        torch.load = original_load


def evaluate_model(
    weights_path: str, data_yaml: str, device: str = "0", img_size: int = 640
) -> dict[str, float]:
    """Evaluate a checkpoint on a dataset's validation split.

    Args:
        weights_path: Path to the ``.pt`` checkpoint to evaluate.
        data_yaml: Path to the dataset descriptor.
        device: Compute device ("0" for the first GPU, or "cpu").
        img_size: Inference image size.

    Returns:
        A mapping of ``precision``, ``recall``, ``mAP50``, and ``mAP50_95``.
    """
    with _trusted_weights_load():
        results = yolo_val.run(
            data=data_yaml,
            weights=weights_path,
            imgsz=img_size,
            device=device,
            task="val",
        )
    precision, recall, map50, map50_95 = (float(x) for x in results[0][:4])
    return {"precision": precision, "recall": recall, "mAP50": map50, "mAP50_95": map50_95}


def calculate_metrics(true_positives: int, false_positives: int, false_negatives: int) -> dict[str, float]:
    """Compute precision, recall, and F1 from detection counts.

    Args:
        true_positives: Count of correct detections.
        false_positives: Count of spurious detections.
        false_negatives: Count of missed objects.

    Returns:
        A mapping of ``precision``, ``recall``, and ``f1`` (each 0 when undefined).
    """
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def generate_confusion_matrix(
    matrix: np.ndarray, class_names: list[str], output_path: Path
) -> Path:
    """Render a confusion matrix as a two-color (abyss-to-teal) heatmap.

    Args:
        matrix: Square confusion-matrix counts.
        class_names: Axis labels, in matrix order.
        output_path: Destination PNG path.

    Returns:
        The written figure path.
    """
    matrix = np.asarray(matrix, dtype=float)
    teal_cmap = _abyss_to_teal_cmap()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap=teal_cmap, aspect="equal")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, color=COLOR_TEXT, fontfamily="monospace", fontsize=8)
    ax.set_yticklabels(class_names, color=COLOR_TEXT, fontfamily="monospace", fontsize=8)
    ax.set_xlabel("Predicted", color=COLOR_TEXT, fontfamily="monospace")
    ax.set_ylabel("Actual", color=COLOR_TEXT, fontfamily="monospace")
    maximum = matrix.max() if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = COLOR_ABYSS if matrix[i, j] > maximum / 2 else COLOR_TEXT
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color=color, fontsize=9)
    fig.savefig(output_path, facecolor=COLOR_ABYSS, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_pr_curve(
    recalls: list[float], precisions: list[float], output_path: Path
) -> Path:
    """Render a precision-recall curve as a single teal line on the abyss.

    Args:
        recalls: Recall values (x-axis), ascending.
        precisions: Precision values (y-axis), aligned with ``recalls``.
        output_path: Destination PNG path.

    Returns:
        The written figure path.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_facecolor(COLOR_ABYSS)
    ax.plot(recalls, precisions, color=COLOR_TEAL, linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color=COLOR_HAIRLINE, linewidth=0.5)
    ax.set_xlabel("Recall", color=COLOR_TEXT, fontfamily="monospace")
    ax.set_ylabel("Precision", color=COLOR_TEXT, fontfamily="monospace")
    ax.tick_params(colors=COLOR_TEXT)
    for spine in ax.spines.values():
        spine.set_color(COLOR_HAIRLINE)
    fig.savefig(output_path, facecolor=COLOR_ABYSS, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_detection_samples(
    samples: list[tuple[np.ndarray, list[dict]]], output_dir: Path
) -> list[Path]:
    """Draw detection boxes onto sample images and save them.

    Args:
        samples: ``(image, detections)`` pairs to render.
        output_dir: Directory to write the annotated samples into.

    Returns:
        The written image paths.
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, (image, detections) in enumerate(samples):
        annotated = draw_boxes(image, detections)
        path = output_dir / f"sample_{index:03d}.png"
        cv2.imwrite(str(path), annotated)
        written.append(path)
    return written


def compare_models(metrics_by_version: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Rank model versions by mAP@0.5 and identify the best.

    Args:
        metrics_by_version: Mapping of version name to its metrics dict (each
            containing at least ``mAP50``).

    Returns:
        A dict with ``ranking`` (versions best-to-worst by mAP50) and ``best``
        (the top version name, or None if no versions were given).
    """
    ranking = sorted(
        metrics_by_version,
        key=lambda version: metrics_by_version[version].get("mAP50", 0.0),
        reverse=True,
    )
    return {"ranking": ranking, "best": ranking[0] if ranking else None}


def _abyss_to_teal_cmap():
    """Build a two-color colormap from the abyss background to teal.

    Returns:
        A Matplotlib ``LinearSegmentedColormap`` from abyss to teal.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("abyss_teal", [COLOR_ABYSS, COLOR_TEAL])
