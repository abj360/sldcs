#!/usr/bin/env python3

"""config.py -- loads and validates the SLDCS runtime configuration.

Defines the :class:`Settings` class, which loads every runtime-tunable value
from the settings YAML, environment variables, and the model registry, validates
them, and exposes them as typed attributes. This module is the single place that
turns configuration sources into a validated configuration object; it performs
no I/O beyond reading those configuration sources and never loads model weights
or handles requests itself.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final

import yaml

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH: Final[Path] = PROJECT_ROOT / "config" / "settings.yaml"
MODEL_REGISTRY_PATH: Final[Path] = PROJECT_ROOT / "weights" / "model_registry.json"

VALID_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


class ConfigurationError(RuntimeError):
    """Raised when the resolved configuration is invalid or incomplete.

    Carries a message identifying the specific setting at fault so a
    misconfiguration is actionable rather than opaque.
    """


class Settings:
    """Validated, typed runtime configuration for the application.

    Responsible for turning the settings YAML, environment overrides, and the
    model registry into a single validated configuration object. It is not
    responsible for loading model weights, opening the log file, or serving
    requests — it only resolves and validates configuration values.

    Attributes:
        MODEL_PATH: Absolute path to the active checkpoint, resolved from the
            model registry (or an explicit ``MODEL_PATH`` environment override).
        CONF_THRESHOLD: Minimum detection confidence to keep (0.0-1.0).
        IOU_THRESHOLD: NMS IoU applied within each tile during inference.
        DEDUP_IOU_THRESHOLD: IoU above which detections from overlapping tiles are
            merged during duplicate removal.
        TILE_SIZE: Square tile edge length in pixels.
        TILE_OVERLAP: Overlap between adjacent tiles in pixels.
        DEVICE: Configured compute device string ("auto", "cpu", or "cuda:N").
        MAX_FILE_SIZE: Maximum accepted upload size in bytes.
        HOST: Server bind host.
        PORT: Server bind port.
        DEBUG: Whether debug behaviour is enabled.
        LOG_LEVEL: Logging verbosity name.
        LOG_FILE: Path to the application log file.
    """

    def __init__(self) -> None:
        """Initialize settings with the built-in defaults.

        The defaults are overwritten by :meth:`load_from_yaml`,
        :meth:`load_from_env`, and model-path resolution before use; call
        :func:`get_settings` for a fully loaded and validated instance.
        """
        self.MODEL_PATH: str = ""
        self.CONF_THRESHOLD: float = 0.40
        self.IOU_THRESHOLD: float = 0.45
        self.DEDUP_IOU_THRESHOLD: float = 0.50
        self.TILE_SIZE: int = 640
        self.TILE_OVERLAP: int = 64
        self.DEVICE: str = "auto"
        self.MAX_FILE_SIZE: int = 52_428_800
        self.HOST: str = "0.0.0.0"
        self.PORT: int = 8000
        self.DEBUG: bool = False
        self.LOG_LEVEL: str = "INFO"
        self.LOG_FILE: str = "logs/sldcs.log"

    def load_from_yaml(self, yaml_path: Path = DEFAULT_SETTINGS_PATH) -> None:
        """Overlay settings from a YAML configuration file.

        Only keys present in the file are applied; missing keys keep their
        current value. Unknown keys are ignored so the YAML can carry
        documentation-only fields without breaking loading.

        Args:
            yaml_path: Path to the settings YAML file.

        Raises:
            ConfigurationError: If the file exists but does not parse to a mapping.
        """
        if not yaml_path.is_file():
            return
        with yaml_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"Settings file {yaml_path} must contain a mapping.")
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def load_from_env(self) -> None:
        """Overlay settings from environment variables.

        Applies ``MODEL_PATH``, ``CONF_THRESHOLD``, ``MAX_FILE_SIZE``,
        ``LOG_LEVEL``, and ``DEBUG`` when present and non-empty. Environment
        values take precedence over the YAML file.
        """
        model_path = os.environ.get("MODEL_PATH")
        if model_path:
            self.MODEL_PATH = model_path
        conf = os.environ.get("CONF_THRESHOLD")
        if conf:
            self.CONF_THRESHOLD = float(conf)
        max_size = os.environ.get("MAX_FILE_SIZE")
        if max_size:
            self.MAX_FILE_SIZE = int(max_size)
        log_level = os.environ.get("LOG_LEVEL")
        if log_level:
            self.LOG_LEVEL = log_level.upper()
        debug = os.environ.get("DEBUG")
        if debug:
            self.DEBUG = debug.strip().lower() in {"1", "true", "yes", "on"}

    def resolve_model_path(self, registry_path: Path = MODEL_REGISTRY_PATH) -> None:
        """Resolve ``MODEL_PATH`` from the model registry if not already set.

        When ``MODEL_PATH`` has not been set explicitly (e.g. via the environment),
        this reads the registry, looks up the ``current_production`` model, and
        uses that entry's ``path`` (resolved against the project root). The
        registry is the single source of truth for the production model path.

        Args:
            registry_path: Path to ``model_registry.json``.

        Raises:
            ConfigurationError: If the registry is missing, names no production
                model, or the production entry has no path.
        """
        if self.MODEL_PATH:
            return
        if not registry_path.is_file():
            raise ConfigurationError(
                f"Model registry not found at {registry_path}. Run "
                "scripts/download_pretrained.py and scripts/setup.py first."
            )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        current = registry.get("current_production")
        if not current:
            raise ConfigurationError(
                "Model registry has no current_production model. Register a model "
                "before starting the application."
            )
        entry = next(
            (model for model in registry.get("models", []) if model.get("version") == current),
            None,
        )
        if entry is None or not entry.get("path"):
            raise ConfigurationError(
                f"Model registry entry for '{current}' is missing a 'path' field."
            )
        resolved = Path(entry["path"])
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        self.MODEL_PATH = str(resolved)

    def validate_settings(self) -> None:
        """Validate the resolved configuration.

        Raises:
            ConfigurationError: If any value is out of range, malformed, or the
                resolved model checkpoint does not exist on disk.
        """
        if not 0.0 <= self.CONF_THRESHOLD <= 1.0:
            raise ConfigurationError(
                f"CONF_THRESHOLD must be within [0, 1], got {self.CONF_THRESHOLD}."
            )
        for name in ("IOU_THRESHOLD", "DEDUP_IOU_THRESHOLD"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{name} must be within [0, 1], got {value}.")
        if self.TILE_SIZE <= 0:
            raise ConfigurationError(f"TILE_SIZE must be positive, got {self.TILE_SIZE}.")
        if not 0 <= self.TILE_OVERLAP < self.TILE_SIZE:
            raise ConfigurationError(
                f"TILE_OVERLAP must be in [0, TILE_SIZE), got {self.TILE_OVERLAP}."
            )
        if self.MAX_FILE_SIZE <= 0:
            raise ConfigurationError(f"MAX_FILE_SIZE must be positive, got {self.MAX_FILE_SIZE}.")
        if not 1 <= self.PORT <= 65535:
            raise ConfigurationError(f"PORT must be within [1, 65535], got {self.PORT}.")
        if self.LOG_LEVEL not in VALID_LOG_LEVELS:
            raise ConfigurationError(
                f"LOG_LEVEL must be one of {sorted(VALID_LOG_LEVELS)}, got {self.LOG_LEVEL}."
            )
        if not self.MODEL_PATH or not Path(self.MODEL_PATH).is_file():
            raise ConfigurationError(
                f"Model checkpoint not found at '{self.MODEL_PATH}'. Run "
                "scripts/download_pretrained.py to fetch it."
            )

    def get_model_config(self) -> dict[str, Any]:
        """Return the configuration needed to construct the inference engine.

        Returns:
            A dict with the model path, device, and every detection/tiling
            threshold the pipeline depends on.
        """
        return {
            "model_path": self.MODEL_PATH,
            "device": self.DEVICE,
            "conf_threshold": self.CONF_THRESHOLD,
            "iou_threshold": self.IOU_THRESHOLD,
            "dedup_iou_threshold": self.DEDUP_IOU_THRESHOLD,
            "tile_size": self.TILE_SIZE,
            "tile_overlap": self.TILE_OVERLAP,
        }


def get_settings(
    yaml_path: Path = DEFAULT_SETTINGS_PATH,
    registry_path: Path = MODEL_REGISTRY_PATH,
) -> Settings:
    """Build a fully loaded and validated :class:`Settings` instance.

    Applies, in precedence order, the built-in defaults, the YAML file, and
    environment overrides; resolves the model path from the registry; then
    validates the result.

    Args:
        yaml_path: Path to the settings YAML file.
        registry_path: Path to the model registry.

    Returns:
        A validated :class:`Settings` instance.

    Raises:
        ConfigurationError: If the resolved configuration is invalid.
    """
    settings = Settings()
    settings.load_from_yaml(yaml_path)
    settings.load_from_env()
    settings.resolve_model_path(registry_path)
    settings.validate_settings()
    return settings
