#!/usr/bin/env bash
# deploy.sh -- builds and launches the SLDCS container stack.
#
# Validates that the environment file and production model are in place, builds
# the image, starts the service via docker compose, and waits for the instrument
# to report ready. Run from anywhere: paths are resolved relative to the repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker/docker-compose.yml"
HEALTH_URL="http://127.0.0.1:8000/ready"

cd "${ROOT_DIR}"

# 1. Environment file must exist (never committed; created from the template).
if [[ ! -f "${ROOT_DIR}/.env" ]]; then
  echo "No .env found; creating one from .env.example. Review it before production use."
  cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"
fi

# 2. A production model must be registered and present on disk.
if [[ ! -f "${ROOT_DIR}/weights/model_registry.json" ]]; then
  echo "ERROR: weights/model_registry.json missing. Run scripts/download_pretrained.py." >&2
  exit 1
fi
if [[ ! -f "${ROOT_DIR}/weights/pretrained/yolov5s.pt" ]]; then
  echo "ERROR: pretrained checkpoint missing. Run scripts/download_pretrained.py." >&2
  exit 1
fi

# 3. Build and start.
echo "Building SLDCS image..."
docker compose -f "${COMPOSE_FILE}" build

echo "Starting SLDCS..."
docker compose -f "${COMPOSE_FILE}" up -d

# 4. Wait for readiness.
echo "Waiting for the instrument to report ready..."
for _ in $(seq 1 60); do
  if curl -sf "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "SLDCS is ready at http://127.0.0.1:8000/"
    exit 0
  fi
  sleep 2
done

echo "ERROR: SLDCS did not become ready in time. Check: docker compose -f ${COMPOSE_FILE} logs" >&2
exit 1
