# SLDCS — Shrimp Larvae Detection and Counting System

SLDCS is a computer-vision instrument built for the University of Arkansas at
Pine Bluff (UAPB) Aquaculture Research Program to automate larval counting for
shrimp hatchery quality control. A full-tray specimen photograph is processed
through a five-stage pipeline — the image is cropped and segmented into
overlapping tiles, an improved YOLOv5 detector scans each tile for larvae, the
per-tile detections are stitched back into the original image, duplicate
detections from overlapping tiles are reconciled into a single count, and the
system returns a final annotated image together with a precise larvae count. The
system currently runs on the stock Ultralytics YOLOv5s COCO-pretrained
checkpoint while the project's own specimen images are being annotated; once a
model has been trained on annotated data, only the checkpoint referenced in
`weights/model_registry.json` changes — the API, UI, and infrastructure stay the
same.

## Installation

The instrument targets an NVIDIA GB10 (aarch64) with CUDA 13.0, and runs on the
stock YOLOv5s COCO-pretrained checkpoint until a project-trained model exists.

```bash
# 1. Environment (use the CUDA 13.0 wheel index on GPU hosts)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cu130

# 2. Bootstrap directories and fetch the pretrained model
python scripts/setup.py
python scripts/download_pretrained.py

# 3. Run the service and open http://localhost:8000/
python main.py
```

Run the tests with `pytest`. See [SETUP.md](SETUP.md) for the full first-run
guide, [DEPLOYMENT.md](DEPLOYMENT.md) for containerized deployment,
[API.md](API.md) for the HTTP API, [ARCHITECTURE.md](ARCHITECTURE.md) for the
system design, and [TRAINING.md](TRAINING.md) for training a project model.

## Documentation index

- [SETUP.md](SETUP.md) — first-run setup
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design and the five-stage pipeline
- [API.md](API.md) — HTTP API reference
- [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) — how to annotate specimens
- [TRAINING.md](TRAINING.md) — training and promoting a model
- [DEPLOYMENT.md](DEPLOYMENT.md) / [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) — deployment
- [MONITORING.md](MONITORING.md) — health, logs, and what to watch
