# SLDCS API

All endpoints are served by the FastAPI application in `app/main.py`. Interactive
docs are available at `/docs` when the server is running.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | The web interface. |
| GET | `/health` | Liveness and whether the model is loaded. |
| GET | `/ready` | 200 when ready to detect, 503 otherwise. |
| GET | `/version` | Application version. |
| GET | `/config` | Live confidence threshold and tile parameters. |
| GET | `/model/info` | Metadata for the active production model. |
| GET | `/model/classes` | Class names the model can detect. |
| POST | `/detect` | Detect and count larvae in one or more images (annotated). |
| POST | `/batch-detect` | Detect and count across a batch (counts only, faster). |

## `POST /detect`

Multipart upload of one or more image files under the field name `files`
(JPG, PNG, or BMP; max 50 MB each).

```bash
curl -F "files=@specimen.jpg" http://localhost:8000/detect
```

Returns a JSON array with one `DetectionResult` per image:

```json
[
  {
    "filename": "specimen.jpg",
    "larvae_count": 247,
    "detections": [
      {"id": 1, "x1": 120.0, "y1": 84.0, "x2": 148.0, "y2": 121.0,
       "confidence": 0.91, "class_id": 0, "class_name": "larvae"}
    ],
    "average_confidence": 0.78,
    "processing_time_ms": 152.4,
    "tiles_scanned": 12,
    "duplicates_merged": 9,
    "image_width": 1920,
    "image_height": 1080,
    "confidence_distribution": [0,0,0,0,3,12,40,88,74,30],
    "annotated_image": "<base64 PNG>"
  }
]
```

## `POST /batch-detect`

Same upload shape; returns aggregate counts only (no annotated images):

```json
{
  "image_count": 4,
  "total_larvae": 883,
  "average_per_image": 220.75,
  "results": [
    {"filename": "a.jpg", "larvae_count": 210, "average_confidence": 0.79}
  ]
}
```

## Error responses

| Status | Meaning |
|---|---|
| 400 | Empty or undecodable image. |
| 413 | File exceeds the configured `MAX_FILE_SIZE`. |
| 415 | Unsupported file type. |
| 503 | Model not loaded (instrument offline). |
