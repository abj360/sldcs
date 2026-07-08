"""Bootstrap the SLDCS runtime scaffolding from a fresh checkout.

This script recreates every working directory the application and pipeline
expect and writes skeleton versions of the two generated configuration files
(``weights/model_registry.json`` and ``data/dataset.yaml``) when, and only when,
they do not already exist. It is safe to run any number of times: it never
overwrites an existing registry or dataset descriptor, so it cannot destroy real
training history or dataset splits.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Final

import yaml

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# Every directory the project expects to exist, relative to the project root.
# This mirrors the directory tree in docs and is the single source of truth for
# what create_project_directories() guarantees.
PROJECT_DIRECTORIES: Final[tuple[str, ...]] = (
    "data/raw",
    "data/processed/images/train",
    "data/processed/images/val",
    "data/processed/images/test",
    "data/processed/labels/train",
    "data/processed/labels/val",
    "data/processed/labels/test",
    "weights/pretrained",
    "weights/v1_baseline",
    "weights/v2_improved",
    "weights/v3_final",
    "training/configs",
    "training/notebooks",
    "results",
    "app",
    "static",
    "tests/fixtures",
    "config",
    "scripts",
    "docker",
    "docs",
    "logs",
    "uploads",
)

REGISTRY_PATH: Final[Path] = PROJECT_ROOT / "weights" / "model_registry.json"
DATASET_YAML_PATH: Final[Path] = PROJECT_ROOT / "data" / "dataset.yaml"

# Dataset descriptor constants. The single detection class is "larvae"; the
# splits live under data/processed and are empty until annotation is complete.
DATASET_CLASS_NAMES: Final[list[str]] = ["larvae"]
DATASET_NUM_CLASSES: Final[int] = 1


def create_project_directories() -> list[Path]:
    """Create every expected project directory, idempotently.

    Iterates over :data:`PROJECT_DIRECTORIES` and ensures each exists. Existing
    directories are left untouched, so this is always safe to call against an
    already-populated repository.

    Returns:
        The absolute paths of every directory that was newly created (empty if
        all already existed).
    """
    created: list[Path] = []
    for relative in PROJECT_DIRECTORIES:
        directory = PROJECT_ROOT / relative
        if not directory.exists():
            created.append(directory)
        directory.mkdir(parents=True, exist_ok=True)
    return created


def initialize_model_registry(registry_path: Path) -> bool:
    """Write an empty model-registry skeleton if none exists yet.

    The registry is the single source of truth for the production model path, so
    this function refuses to overwrite an existing file — doing so would discard
    real model history. When absent, it writes a skeleton with no registered
    models and no production selection; the pretrained checkpoint is registered
    later, once it has been downloaded and verified.

    Args:
        registry_path: Destination path for ``model_registry.json``.

    Returns:
        ``True`` if a new skeleton was written, ``False`` if a registry already
        existed and was left untouched.
    """
    if registry_path.exists():
        return False
    skeleton = {
        "models": [],
        "current_production": None,
        "last_updated": date.today().isoformat(),
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(skeleton, indent=2) + "\n", encoding="utf-8")
    return True


def initialize_dataset_yaml(dataset_yaml_path: Path) -> bool:
    """Write the dataset descriptor skeleton if none exists yet.

    Writes a YOLO-style dataset descriptor pointing at the processed train/val/
    test splits. Never overwrites an existing descriptor, since the human may
    have already tuned split paths. The split directories are expected to be
    empty until annotation is complete.

    Args:
        dataset_yaml_path: Destination path for ``dataset.yaml``.

    Returns:
        ``True`` if a new descriptor was written, ``False`` if one already
        existed and was left untouched.
    """
    if dataset_yaml_path.exists():
        return False
    descriptor = {
        "path": "data/processed",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": DATASET_NUM_CLASSES,
        "names": DATASET_CLASS_NAMES,
    }
    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(descriptor, handle, sort_keys=False)
    return True


def main() -> None:
    """Run the full bootstrap and print a human-readable summary.

    Creates directories, then initializes the registry and dataset descriptor
    skeletons, reporting for each whether it was created or already present.
    """
    created_dirs = create_project_directories()
    registry_created = initialize_model_registry(REGISTRY_PATH)
    dataset_created = initialize_dataset_yaml(DATASET_YAML_PATH)

    print("SLDCS setup complete.")
    print(f"  Directories created: {len(created_dirs)} (of {len(PROJECT_DIRECTORIES)} expected)")
    for directory in created_dirs:
        print(f"    + {directory.relative_to(PROJECT_ROOT)}")
    print(f"  Model registry:      {'created' if registry_created else 'already present'}")
    print(f"  Dataset descriptor:  {'created' if dataset_created else 'already present'}")


if __name__ == "__main__":
    main()
