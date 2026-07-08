# SLDCS application image.
# Base is a CUDA 13.0 runtime for the project's NVIDIA GB10 (aarch64) hardware.
# On a CPU-only host, swap this base for a plain python image and install the
# CPU torch wheels instead (see docs/DEPLOYMENT.md).
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libraries: Python, plus the shared libs OpenCV needs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip python3.12-venv \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so the layer caches across code changes.
# The CUDA 13.0 aarch64 torch/torchvision wheels come from the PyTorch index.
COPY requirements.txt ./
RUN python3 -m pip install --break-system-packages -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130

# Application code and its static/configuration/model assets.
COPY app ./app
COPY static ./static
COPY config ./config
COPY scripts ./scripts
COPY training ./training
COPY weights ./weights
COPY main.py ./main.py

# Runtime output directories (also provided as volumes in docker-compose).
RUN mkdir -p logs uploads

EXPOSE 8000

# Fail the container's health check if the model has not loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready').status==200 else 1)" || exit 1

CMD ["python3", "main.py"]
