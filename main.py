#!/usr/bin/env python3

"""main.py -- launches the SLDCS web service.

Configures logging from ``config/logging.yaml``, loads and validates the runtime
settings, and starts the uvicorn server bound to the configured host and port.
Run it with ``python main.py``.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Final

import uvicorn
import yaml

from app.config import get_settings

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
LOGGING_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "config" / "logging.yaml"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"


def configure_logging(config_path: Path = LOGGING_CONFIG_PATH) -> None:
    """Apply the logging configuration from a YAML file.

    Ensures the log directory exists first, then applies the dictConfig. Falls
    back to a basic console configuration if the file is absent.

    Args:
        config_path: Path to the logging YAML configuration.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            logging.config.dictConfig(yaml.safe_load(handle))
    else:
        logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Configure logging and run the uvicorn server from validated settings."""
    configure_logging()
    settings = get_settings()
    logging.getLogger("sldcs").info(
        "Starting SLDCS on %s:%s (device target: %s)",
        settings.HOST,
        settings.PORT,
        settings.DEVICE,
    )
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
