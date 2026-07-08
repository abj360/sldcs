# SLDCS training

This document describes how to train a project-specific larvae detector once the
specimen images have been annotated. Until then, the system runs on the stock
pretrained checkpoint and no training is required.

## Prerequisites

1. Raw images imported (`scripts/google_drive_sync.py`) and prepared
   (`scripts/prepare_dataset.py`).
2. Every image annotated in YOLO format (see `docs/ANNOTATION_GUIDE.md`).
3. Annotations validated:
   ```bash
   python scripts/validate_annotations.py
   ```
4. Data split into train/val/test:
   ```bash
   python scripts/prepare_train_data.py
   ```

## Training a version

Each version has a config under `training/configs/`. Train one with the
`ModelTrainer`:

```python
from pathlib import Path
from training.train import ModelTrainer

trainer = ModelTrainer(Path("training/configs/config_v1.yaml"))
metrics = trainer.train()      # runs YOLOv5 training on the project dataset
trainer.validate()             # metrics on the val split
trainer.save_checkpoint()      # copies best.pt into weights/v1_baseline/
trainer.log_metrics(metrics)   # writes weights/v1_baseline/metrics.json
trainer.generate_report()      # writes weights/v1_baseline/training_report.md
```

### Configurations

| Config | Intent |
|---|---|
| `config_v1.yaml` | Baseline transfer learning from COCO-pretrained YOLOv5s. |
| `config_v2.yaml` | Stronger augmentation, longer cosine schedule, larger image size. |
| `config_v3.yaml` | Final candidate at tile resolution with a tuned schedule. |

## Evaluating and comparing

Use `training/evaluate.py` to measure a checkpoint and render figures
(confusion matrix, PR curve, detection samples), and `compare_models` to rank
versions by mAP@0.5. The `training/notebooks/` provide interactive analysis.

## Promoting a model to production

1. Choose the version with the best val mAP@0.5.
2. Add its entry to `weights/model_registry.json` with a real `path`, and set
   `current_production` to that version.
3. Commit the registry change with a `model` commit whose body records the
   config used, the resulting mAP@0.5, and candidate/rejected status, e.g.:

   ```
   model(registry): promote v3_final to production

   Config: training/configs/config_v3.yaml
   mAP@0.5: 0.xx (val)
   Status: candidate accepted; supersedes pretrained_yolov5s
   ```

The API, UI, and infrastructure do not change — only the checkpoint the registry
points at.

## Reproducing the code verification

`train.py` and `evaluate.py` were verified once against a small disposable
dataset to prove correctness; that output was discarded and never committed. No
metrics for `v1_baseline`, `v2_improved`, or `v3_final` exist until you train
them.
