"""YOLOv5 detection engine for SLDCS.

Defines :class:`YOLOv5Inference`, which loads exactly one YOLOv5 checkpoint on
construction and runs the full tile-based detection pipeline for the lifetime of
the process. This is the only module permitted to import ``torch`` or the YOLOv5
model API; it performs no HTTP work and builds no API responses.
"""

from __future__ import annotations

import contextlib
import functools
from pathlib import Path
from typing import Any, Final, Iterator

import numpy as np
import torch
import yolov5

from app.utils import (
    crop_and_tile_image,
    remove_duplicate_detections,
    stitch_results,
)

DEFAULT_INFERENCE_SIZE: Final[int] = 640


@contextlib.contextmanager
def _trusted_weights_load() -> Iterator[None]:
    """Temporarily allow full (non-``weights_only``) checkpoint deserialization.

    PyTorch 2.6 made ``torch.load`` default to ``weights_only=True``, which
    rejects the pickled model classes in the official YOLOv5 checkpoint. The
    checkpoint is the Ultralytics release fetched and SHA-256 verified by
    ``scripts/download_pretrained.py``, so full deserialization is trusted. The
    original ``torch.load`` is restored on exit.

    Yields:
        None.
    """
    original_load = torch.load
    torch.load = functools.partial(original_load, weights_only=False)
    try:
        yield
    finally:
        torch.load = original_load


class YOLOv5Inference:
    """Tile-based YOLOv5 detector for full-tray specimen images.

    Loads one checkpoint at construction and reuses it for every request. Given
    an image it runs the pipeline's detection stages — tile, detect per tile,
    stitch back to the original frame, and remove cross-tile duplicates — and
    returns detections in original-image coordinates. It does not draw
    annotations, encode images, or build HTTP responses; those are the caller's
    responsibility.

    Attributes:
        model_path: Filesystem path of the loaded checkpoint.
        device: Resolved compute device string (e.g. "cuda:0" or "cpu").
        conf_threshold: Minimum confidence for a detection to be kept.
        iou_threshold: NMS IoU applied within each tile.
        dedup_iou_threshold: IoU above which cross-tile detections are merged.
        tile_size: Square tile edge length in pixels.
        tile_overlap: Overlap between adjacent tiles in pixels.
        model: The loaded YOLOv5 model object.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        dedup_iou_threshold: float = 0.50,
        tile_size: int = 640,
        tile_overlap: int = 64,
    ) -> None:
        """Load the checkpoint and configure detection thresholds.

        Args:
            model_path: Path to the YOLOv5 ``.pt`` checkpoint.
            device: "auto" (CUDA if available else CPU), "cpu", or "cuda:N".
            conf_threshold: Minimum detection confidence to keep.
            iou_threshold: NMS IoU applied within each tile.
            dedup_iou_threshold: IoU above which cross-tile detections merge.
            tile_size: Square tile edge length in pixels.
            tile_overlap: Overlap between adjacent tiles in pixels.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
        """
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.dedup_iou_threshold = dedup_iou_threshold
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        with _trusted_weights_load():
            self.model = yolov5.load(model_path, device=self.device)
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve an "auto" device request to a concrete device string.

        Args:
            device: Requested device ("auto", "cpu", or "cuda:N").

        Returns:
            "cuda:0" when ``device`` is "auto" and CUDA is available, "cpu" when
            "auto" without CUDA, otherwise ``device`` unchanged.
        """
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Coerce an input image into the array form the model expects.

        Args:
            image: Input image as an ``(H, W, C)`` array.

        Returns:
            A contiguous ``uint8`` 3-channel BGR array.

        Raises:
            ValueError: If the image is not a 3-channel array.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel (H, W, 3) image.")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return np.ascontiguousarray(image)

    def postprocess_results(self, results: Any) -> list[dict]:
        """Convert one tile's raw YOLOv5 output into detection dicts.

        Args:
            results: The object returned by calling the model on one tile.

        Returns:
            Tile-local detection dicts with ``x1``/``y1``/``x2``/``y2``,
            ``confidence``, ``class_id``, and ``class_name`` keys.
        """
        predictions = results.pred[0]
        names = results.names
        detections: list[dict] = []
        for *xyxy, confidence, class_id in predictions.tolist():
            class_index = int(class_id)
            detections.append(
                {
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                    "confidence": float(confidence),
                    "class_id": class_index,
                    "class_name": names.get(class_index, str(class_index))
                    if isinstance(names, dict)
                    else names[class_index],
                }
            )
        return detections

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        """Run the full tile-based detection pipeline on one image.

        Tiles the image, detects on each tile, stitches detections back into the
        original frame, and removes cross-tile duplicates.

        Args:
            image: Input image as an ``(H, W, C)`` BGR array.

        Returns:
            A dict with ``detections`` (list of dicts in original coordinates),
            ``tiles_scanned`` (int), and ``duplicates_merged`` (int).
        """
        prepared = self.preprocess_image(image)
        height, width = prepared.shape[:2]
        tiles, origins = crop_and_tile_image(prepared, self.tile_size, self.tile_overlap)

        tiles_results: list[list[dict]] = []
        for tile in tiles:
            raw = self.model(tile, size=self.tile_size)
            tiles_results.append(self.postprocess_results(raw))

        stitched = stitch_results(tiles_results, origins, (width, height))
        deduped = remove_duplicate_detections(stitched, self.dedup_iou_threshold)
        return {
            "detections": deduped,
            "tiles_scanned": len(tiles),
            "duplicates_merged": len(stitched) - len(deduped),
        }

    def detect_batch(self, images: list[np.ndarray]) -> list[dict[str, Any]]:
        """Run :meth:`detect` on each image in a batch.

        Args:
            images: Input images as BGR arrays.

        Returns:
            One :meth:`detect` result dict per input image, in order.
        """
        return [self.detect(image) for image in images]

    def get_available_classes(self) -> list[str]:
        """Return the class names the loaded model can detect.

        Returns:
            The model's class names in class-id order.
        """
        names = self.model.names
        if isinstance(names, dict):
            return [names[key] for key in sorted(names)]
        return list(names)

    def get_metadata(self) -> dict[str, Any]:
        """Return runtime metadata describing the loaded model.

        Returns:
            A dict with ``model_path``, ``device``, ``class_count``, and
            ``class_names``.
        """
        classes = self.get_available_classes()
        return {
            "model_path": self.model_path,
            "device": self.device,
            "class_count": len(classes),
            "class_names": classes,
        }
