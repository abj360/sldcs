# SLDCS monitoring

The instrument is a single service; monitoring it is correspondingly simple.

## Health and readiness

| Endpoint | Use |
|---|---|
| `GET /health` | Liveness. Returns `status`, `model_loaded`, `device`, `version`. Use for uptime checks. |
| `GET /ready` | Readiness. Returns 200 only when the model is loaded; 503 otherwise. Use for load-balancer and container health checks. |

The container's `HEALTHCHECK` already polls `/ready`; `docker ps` shows the
health state.

## Logs

Logging is configured by `config/logging.yaml`. Lines go to stdout and to a
rotating file at `logs/sldcs.log` (10 MB × 5 backups). Watch for:

- `Detection model loaded on <device>` at startup — confirms the model is up.
- `Failed to load detection model` — the instrument is offline; the UI shows the
  coral offline state and `/ready` returns 503.

```bash
docker compose -f docker/docker-compose.yml logs -f      # container
tail -f logs/sldcs.log                                    # host / bare-metal
```

## What to watch

- **Model status:** the header status dot (teal = ready, amber = loading, coral
  = offline) mirrors `/health`.
- **Latency:** each `DetectionResult` reports `processing_time_ms`; the UI shows
  a session average. A sustained rise suggests GPU contention or oversized
  uploads.
- **GPU:** `nvidia-smi` for memory and utilization on the host.
- **Disk:** `logs/` and `uploads/` grow over time; logs self-rotate, but prune
  `uploads/` periodically.

## There is no hidden failure mode for counts

A zero count is a real, reported result — it is never suppressed or replaced
with a friendly message. If counts look wrong, inspect the annotated image and
the per-detection confidences rather than assuming a service fault.
