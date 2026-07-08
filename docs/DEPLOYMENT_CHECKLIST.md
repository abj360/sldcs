# SLDCS deployment checklist

Work through this before putting the instrument in front of hatchery staff.

## Environment

- [ ] Git identity set for this repository (see `docs/SETUP.md`).
- [ ] `.env` created from `.env.example` and reviewed (no placeholder values left).
- [ ] GPU host: NVIDIA driver + Container Toolkit installed; `nvidia-smi` works.
- [ ] `python -c "import torch; assert torch.cuda.is_available()"` passes on the host (or CPU fallback intentionally chosen).

## Model

- [ ] `weights/pretrained/yolov5s.pt` present and its SHA-256 matches `weights/pretrained/metadata.json`.
- [ ] `weights/model_registry.json` `current_production` points at an existing checkpoint.
- [ ] `scripts/download_pretrained.py` load test passes.

## Application

- [ ] `pytest` passes.
- [ ] `python main.py` starts and `GET /health` reports `model_loaded: true`.
- [ ] `GET /model/info` shows the intended model and honest training status.
- [ ] A sample image through `POST /detect` returns a plausible count and an annotated image.

## Container

- [ ] `docker compose -f docker/docker-compose.yml build` succeeds.
- [ ] `docker compose -f docker/docker-compose.yml up -d` starts the service.
- [ ] Container health check reaches `ready` within the start period.
- [ ] `logs/`, `uploads/`, and `weights/` volumes mount and persist across restarts.

## Post-deploy

- [ ] Confirm the UI loads and the status dot is teal.
- [ ] Confirm logs are being written to `logs/sldcs.log`.
- [ ] Record the deployed version (`GET /version`) in the operations log.
