#!/usr/bin/env python3

"""train.py -- trains one version of the SLDCS larvae detector.

Defines :class:`ModelTrainer`, which runs a single complete YOLOv5 training run
for one configuration version, records its metrics, and copies the resulting
best checkpoint into that version's weights directory. It is intentionally
independent of the running application: it never imports ``app.config`` and knows
nothing about the FastAPI service.
"""

from __future__ import annotations

import contextlib
import csv
import functools
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterator

import torch
import yaml
import yolov5.train as yolo_train
import yolov5.val as yolo_val

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
RUNS_DIR: Final[Path] = PROJECT_ROOT / "runs" / "train"
MAP_METRIC_KEY: Final[str] = "metrics/mAP_0.5"


@contextlib.contextmanager
def _trusted_weights_load() -> Iterator[None]:
    """Temporarily allow full deserialization of the trusted base checkpoint.

    Training starts from the SHA-verified official YOLOv5 checkpoint, which
    PyTorch 2.6+ refuses to load under the ``weights_only=True`` default. Restores
    the original ``torch.load`` on exit.

    Yields:
        None.
    """
    original_load = torch.load
    torch.load = functools.partial(original_load, weights_only=False)
    try:
        yield
    finally:
        torch.load = original_load


class ModelTrainer:
    """Run and record one YOLOv5 training version.

    Owns the full lifecycle of a single training run: reading its configuration,
    validating the dataset is ready, invoking YOLOv5 training, reading back the
    metrics, saving the best checkpoint into the version's weights directory, and
    writing a training report. It does not register the model in the registry or
    touch the application — those are separate, deliberate steps.

    Attributes:
        version: The version identifier (e.g. "v1_baseline").
        config: The parsed training configuration.
        model: Path to the best checkpoint after training, or None before.
        device: Resolved compute device string.
        epoch: Number of epochs configured for the run.
        best_metrics: Best validation metrics recorded during the run.
        training_history: Per-epoch metric rows read from results.csv.
        start_time: UTC timestamp when the run began, or None before.
    """

    def __init__(self, config_path: Path, overrides: dict[str, Any] | None = None) -> None:
        """Load a training configuration and prepare trainer state.

        Args:
            config_path: Path to a ``config_vN.yaml`` file.
            overrides: Optional config keys to override (used to point a
                verification run at a small example dataset).
        """
        with Path(config_path).open("r", encoding="utf-8") as handle:
            self.config: dict[str, Any] = yaml.safe_load(handle)
        if overrides:
            self.config.update(overrides)
        self.version: str = self.config["version"]
        self.device: str = self._resolve_device(self.config.get("device", "auto"))
        self.epoch: int = int(self.config["epochs"])
        self.model: str | None = None
        self.best_metrics: dict[str, float] = {}
        self.training_history: list[dict[str, str]] = []
        self.start_time: datetime | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve an "auto" device request to a concrete device.

        Args:
            device: "auto", "cpu", or "cuda:N".

        Returns:
            The resolved device string. YOLOv5 expects the CUDA ordinal alone
            (e.g. "0"), so "cuda:0" is normalized to "0".
        """
        if device == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return device.replace("cuda:", "") if device.startswith("cuda:") else device

    def prepare_data(self) -> str:
        """Validate the dataset descriptor and return its path.

        For the project's own dataset, confirms the descriptor exists and the
        train/val split directories contain images (i.e. annotation and
        splitting have been done). External example datasets (used for
        verification) are passed through without the emptiness check.

        Returns:
            The dataset descriptor path to hand to YOLOv5.

        Raises:
            FileNotFoundError: If the dataset descriptor does not exist.
            RuntimeError: If the project dataset's train/val splits are empty.
        """
        dataset = self.config["dataset"]
        candidate = Path(dataset)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / dataset
        is_project_dataset = candidate == (PROJECT_ROOT / "data" / "dataset.yaml")
        if not is_project_dataset:
            # An external/example dataset (e.g. the bundled coco128.yaml used for
            # verification): pass the name through for YOLOv5's own resolver.
            return dataset
        if not candidate.is_file():
            raise FileNotFoundError(f"Dataset descriptor not found: {candidate}")
        descriptor = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        base = PROJECT_ROOT / descriptor.get("path", "data/processed")
        for split_key in ("train", "val"):
            split_dir = base / descriptor[split_key]
            if not split_dir.is_dir() or not any(split_dir.iterdir()):
                raise RuntimeError(
                    f"Split '{split_key}' at {split_dir} is empty. Annotate and "
                    "run scripts/prepare_train_data.py before training."
                )
        return str(candidate)

    def train(self) -> dict[str, float]:
        """Execute the YOLOv5 training run and record its metrics.

        Returns:
            The best validation metrics from the run (empty if no results.csv
            was produced).
        """
        self.start_time = datetime.now(timezone.utc)
        data_path = self.prepare_data()
        base_weights = self.config["base_weights"]
        if not Path(base_weights).is_absolute():
            base_weights = str(PROJECT_ROOT / base_weights)

        kwargs = {
            "data": data_path,
            "weights": base_weights,
            "epochs": self.epoch,
            "batch_size": int(self.config["batch_size"]),
            "imgsz": int(self.config["img_size"]),
            "device": self.device,
            "workers": int(self.config.get("workers", 8)),
            "patience": int(self.config.get("patience", 100)),
            "optimizer": self.config.get("optimizer", "SGD"),
            "cos_lr": bool(self.config.get("cos_lr", False)),
            "project": str(RUNS_DIR),
            "name": self.version,
            "exist_ok": True,
        }
        with _trusted_weights_load():
            yolo_train.run(**kwargs)

        save_dir = RUNS_DIR / self.version
        self.model = str(save_dir / "weights" / "best.pt")
        self.best_metrics = self._read_results(save_dir / "results.csv")
        return self.best_metrics

    def _read_results(self, results_csv: Path) -> dict[str, float]:
        """Read per-epoch metrics from a YOLOv5 results.csv.

        Args:
            results_csv: Path to the run's results.csv.

        Returns:
            The row with the highest mAP@0.5 as a float mapping; empty if the
            file is absent.
        """
        if not results_csv.is_file():
            return {}
        with results_csv.open("r", encoding="utf-8") as handle:
            rows = [
                {key.strip(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        self.training_history = rows
        if not rows:
            return {}
        best_row = max(rows, key=lambda r: float(r.get(MAP_METRIC_KEY, 0.0) or 0.0))
        return {key: float(value) for key, value in best_row.items() if _is_float(value)}

    def validate(self) -> dict[str, float]:
        """Validate the trained best checkpoint on the dataset's val split.

        Returns:
            A mapping of ``precision``, ``recall``, ``mAP50``, and ``mAP50_95``.

        Raises:
            RuntimeError: If called before a checkpoint has been produced.
        """
        if not self.model or not Path(self.model).is_file():
            raise RuntimeError("No trained checkpoint to validate; run train() first.")
        with _trusted_weights_load():
            results = yolo_val.run(
                data=self.prepare_data(),
                weights=self.model,
                imgsz=int(self.config["img_size"]),
                batch_size=int(self.config["batch_size"]),
                device=self.device,
                task="val",
            )
        metrics_tuple = results[0]
        precision, recall, map50, map50_95 = (float(x) for x in metrics_tuple[:4])
        return {"precision": precision, "recall": recall, "mAP50": map50, "mAP50_95": map50_95}

    def save_checkpoint(self) -> Path:
        """Copy the run's best checkpoint into the version's weights directory.

        Returns:
            The destination path of the copied checkpoint.

        Raises:
            RuntimeError: If there is no best checkpoint to copy.
        """
        if not self.model or not Path(self.model).is_file():
            raise RuntimeError("No trained checkpoint to save; run train() first.")
        output_dir = PROJECT_ROOT / self.config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "best.pt"
        shutil.copy2(self.model, destination)
        return destination

    def log_metrics(self, metrics: dict[str, float]) -> Path:
        """Write the recorded metrics to the version's metrics.json.

        Args:
            metrics: Metrics to persist.

        Returns:
            The path of the written metrics.json.
        """
        output_dir = PROJECT_ROOT / self.config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        payload = {
            "version": self.version,
            "metrics": metrics,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return metrics_path

    def generate_report(self) -> Path:
        """Write a Markdown training report for the run.

        Returns:
            The path of the written report.
        """
        output_dir = PROJECT_ROOT / self.config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "training_report.md"
        lines = [
            f"# Training report — {self.version}",
            "",
            f"- Description: {self.config.get('description', '')}",
            f"- Base weights: {self.config['base_weights']}",
            f"- Epochs: {self.epoch}",
            f"- Image size: {self.config['img_size']}",
            f"- Device: {self.device}",
            f"- Started: {self.start_time.isoformat() if self.start_time else 'n/a'}",
            "",
            "## Best metrics",
            "",
        ]
        if self.best_metrics:
            for key, value in self.best_metrics.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("No metrics recorded.")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path


def _is_float(value: str) -> bool:
    """Return whether a string parses as a float.

    Args:
        value: Candidate string.

    Returns:
        True if ``float(value)`` succeeds.
    """
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
