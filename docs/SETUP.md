# SLDCS setup

This guide takes a fresh clone to a running instrument.

## 1. Configure the git identity (first-time, this repo only)

This repository ships with a placeholder commit identity. Set your real
identity for this repository before committing:

```bash
git config user.name  "Your Name"
git config user.email "you@uapb.edu"
```

(If a real `user.name`/`user.email` was already configured in your environment,
the build reused it and you can skip this.)

## 2. Create the environment

The project targets the NVIDIA GB10 (aarch64) with CUDA 13.0. Create a virtual
environment and install the pinned dependencies, using the PyTorch CUDA 13.0
wheel index for the GPU builds:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cu130
python -c "import torch; assert torch.cuda.is_available()"
```

On a CPU-only machine, install the matching CPU wheels instead (drop the
`--extra-index-url` and install the CPU `torch`/`torchvision`).

## 3. Bootstrap directories and the pretrained model

```bash
python scripts/setup.py               # create runtime dirs + registry/dataset skeletons
python scripts/download_pretrained.py # fetch and verify weights/pretrained/yolov5s.pt
```

### About the tracked model weight

`weights/pretrained/yolov5s.pt` is ~14 MB and is tracked in git (well under
common hosting limits), so a fresh clone already has a working model. If your
git host rejects it, remove it from tracking and fetch it on first setup with
`python scripts/download_pretrained.py` instead — the registry path is unchanged
either way.

## 4. Import and prepare the specimen data (when ready)

The very first data command is the Google Drive sync:

```bash
python scripts/google_drive_sync.py --folder-id <DRIVE_FOLDER_ID>
python scripts/validate_data.py
python scripts/prepare_dataset.py
```

## 5. Run the application

```bash
python main.py            # serves on http://0.0.0.0:8000
```

Copy `.env.example` to `.env` first if you want to override any defaults.

## 6. Run the tests

```bash
pytest
```
