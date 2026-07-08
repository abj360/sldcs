#!/usr/bin/env python3

"""main.py -- assembles and exposes the FastAPI application.

Wires the HTTP application together: instantiates the app, configures CORS,
mounts the static UI, constructs the single shared detection engine at startup,
and registers the API routes. All request-handling logic lives in
:mod:`app.routes`; this module only assembles and exposes the application.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.config import PROJECT_ROOT, Settings, get_settings
from app.inference import ModelInference
from app.models import (
    BatchDetectionResponse,
    DetectionResult,
    HealthCheckResponse,
    ModelInfoResponse,
)
from app.routes import (
    _decode_image_bytes,
    enforce_upload_count,
    handle_batch_detection,
    handle_detection,
    validate_upload,
)

STATIC_DIR: Final[Path] = PROJECT_ROOT / "static"
MODEL_REGISTRY_PATH: Final[Path] = PROJECT_ROOT / "weights" / "model_registry.json"
LOGGER: Final[logging.Logger] = logging.getLogger("sldcs")

app = FastAPI(title="SLDCS — Shrimp Larvae Detection and Counting System", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_inference() -> ModelInference:
    """Return the loaded detection engine or fail with a 503.

    Returns:
        The process-wide :class:`ModelInference` instance.

    Raises:
        HTTPException: 503 if the model failed to load at startup.
    """
    engine = getattr(app.state, "inference", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Instrument offline — detection model failed to load.",
        )
    return engine


def get_active_settings() -> Settings:
    """Return the settings loaded at startup.

    Returns:
        The process-wide :class:`Settings` instance.
    """
    return app.state.settings


@app.on_event("startup")
def _startup() -> None:
    """Load settings and construct the shared detection engine.

    A model-load failure is logged and leaves the engine unset so the UI can
    render its offline state rather than the whole service failing to boot.
    """
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO))
    app.state.settings = settings
    try:
        app.state.inference = ModelInference(**settings.get_model_config())
        LOGGER.info("Detection model loaded on %s.", app.state.inference.device)
    except Exception as error:  # noqa: BLE001 - degrade to offline, do not crash boot
        app.state.inference = None
        LOGGER.error("Failed to load detection model: %r", error)


@app.on_event("shutdown")
def _shutdown() -> None:
    """Release the detection engine reference on shutdown."""
    app.state.inference = None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page UI.

    Returns:
        The static ``index.html`` page.
    """
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health", response_model=HealthCheckResponse)
def health() -> HealthCheckResponse:
    """Report liveness and whether the model is loaded.

    Returns:
        The current health status.
    """
    engine = getattr(app.state, "inference", None)
    return HealthCheckResponse(
        status="ok" if engine is not None else "unavailable",
        model_loaded=engine is not None,
        device=engine.device if engine is not None else "none",
        version=__version__,
    )


@app.get("/ready")
def ready() -> JSONResponse:
    """Report readiness to serve detections.

    Returns:
        A 200 response when the model is loaded, otherwise 503.
    """
    engine = getattr(app.state, "inference", None)
    if engine is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"ready": False}
        )
    return JSONResponse(content={"ready": True})


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info(engine: ModelInference = Depends(get_inference)) -> ModelInfoResponse:
    """Return metadata describing the active production model.

    Reproduces the registry entry for the current production model and augments
    it with the loaded model's device and class information.

    Args:
        engine: The loaded detection engine (injected).

    Returns:
        The model info response.
    """
    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    current = registry.get("current_production")
    entry = next(
        (model for model in registry.get("models", []) if model.get("version") == current),
        {},
    )
    metadata = engine.get_metadata()
    trained = bool(entry.get("trained_on_project_data", False))
    note = (
        "This is the stock COCO-pretrained model checkpoint. No model has been "
        "trained on project specimen data yet; no project mAP is available."
        if not trained
        else "This model was trained on annotated SLDCS specimen data."
    )
    return ModelInfoResponse(
        version=entry.get("version", "unknown"),
        source=entry.get("source", "unknown"),
        trained_on_project_data=trained,
        status=entry.get("status", "unknown"),
        device=metadata["device"],
        class_count=metadata["class_count"],
        class_names=metadata["class_names"],
        note=note,
    )


@app.get("/model/classes", response_model=list[str])
def model_classes(engine: ModelInference = Depends(get_inference)) -> list[str]:
    """Return the class names the loaded model can detect.

    Args:
        engine: The loaded detection engine (injected).

    Returns:
        The model's class names.
    """
    return engine.get_available_classes()


@app.get("/version")
def version() -> JSONResponse:
    """Return the application version.

    Returns:
        A JSON object with the version string.
    """
    return JSONResponse(content={"version": __version__})


@app.get("/config")
def runtime_config(settings: Settings = Depends(get_active_settings)) -> JSONResponse:
    """Return the live tiling and confidence parameters for the UI.

    Args:
        settings: Active settings (injected).

    Returns:
        A JSON object with the confidence threshold, tile size, and tile overlap.
    """
    return JSONResponse(
        content={
            "conf_threshold": settings.CONF_THRESHOLD,
            "tile_size": settings.TILE_SIZE,
            "tile_overlap": settings.TILE_OVERLAP,
        }
    )


@app.post("/detect", response_model=list[DetectionResult])
async def detect(
    files: list[UploadFile] = File(...),
    engine: ModelInference = Depends(get_inference),
    settings: Settings = Depends(get_active_settings),
) -> list[DetectionResult]:
    """Detect and count larvae in one or more uploaded images (annotated).

    Args:
        files: One or more uploaded specimen images.
        engine: The loaded detection engine (injected).
        settings: Active settings (injected).

    Returns:
        One annotated detection result per uploaded image.
    """
    enforce_upload_count(len(files))
    images = []
    for upload in files:
        data = await validate_upload(upload, settings.MAX_FILE_SIZE)
        images.append((upload.filename, _decode_image_bytes(data)))
    return await run_in_threadpool(handle_detection, images, engine, settings)


@app.post("/batch-detect", response_model=BatchDetectionResponse)
async def batch_detect(
    files: list[UploadFile] = File(...),
    engine: ModelInference = Depends(get_inference),
    settings: Settings = Depends(get_active_settings),
) -> BatchDetectionResponse:
    """Detect and count larvae across a batch, returning counts only.

    Args:
        files: The uploaded specimen images.
        engine: The loaded detection engine (injected).
        settings: Active settings (injected).

    Returns:
        The aggregate batch response.
    """
    enforce_upload_count(len(files))
    images = []
    for upload in files:
        data = await validate_upload(upload, settings.MAX_FILE_SIZE)
        images.append((upload.filename, _decode_image_bytes(data)))
    return await run_in_threadpool(handle_batch_detection, images, engine, settings)
