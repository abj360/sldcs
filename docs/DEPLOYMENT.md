# SLDCS deployment

The application ships as a container. The image bundles the app, the static UI,
the configuration, and the model weights; runtime output and the weights
directory are mounted as volumes so a model can be swapped without a rebuild.

## Prerequisites

- Docker with the Compose plugin.
- On GPU hosts: the NVIDIA Container Toolkit and a driver compatible with
  CUDA 13.0 (the project's GB10 target).
- `weights/pretrained/yolov5s.pt` present (`python scripts/download_pretrained.py`).

## One-command deploy

```bash
scripts/deploy.sh
```

The script verifies the environment file and the production model, builds the
image, starts the service, and waits for `/ready`.

## Manual deploy

```bash
cp .env.example .env            # then edit as needed
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
curl http://localhost:8000/ready
```

## GPU vs CPU

The Dockerfile bases on a CUDA 13.0 runtime and the Compose file reserves all
NVIDIA GPUs. On a CPU-only host:

1. Change the Dockerfile base image to a plain `python:3.12-slim` image.
2. Install CPU `torch`/`torchvision` wheels (drop the CUDA extra index URL).
3. Remove the `deploy.resources.reservations.devices` block from the Compose
   file.

`DEVICE=auto` in the settings resolves to CPU automatically when CUDA is absent.

## Configuration

All runtime configuration is environment-driven; see `.env.example` for the full
list. The active model is resolved from `weights/model_registry.json`, never a
hardcoded path.

## Volumes

| Host path | Container path | Purpose |
|---|---|---|
| `logs/` | `/app/logs` | Rotating application logs. |
| `uploads/` | `/app/uploads` | Transient upload scratch space. |
| `weights/` | `/app/weights` | Model checkpoints and the registry (swap models here). |
