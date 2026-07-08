# SLDCS — Shrimp Larvae Detection and Counting System

SLDCS is a computer-vision instrument built by the University of Arkansas at
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

(to be completed once the application is built)
