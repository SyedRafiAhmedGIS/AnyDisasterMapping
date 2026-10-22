# ---------------------------------------------------------------------------
# AnyDisasterMapping — CPU/GPU-ready container
#
#   Build (CPU / GPU alike — torch wheels bundle the CUDA runtime):
#     docker build -t anydisaster .
#
#   GPU (requires nvidia-container-toolkit):
#     docker run --rm --gpus all anydisaster python scripts/smoke_test_all_domains.py
#
#   Gradio demo app (separate target):
#     docker build -t anydisaster-demo --target demo .
#     docker run --rm -p 7860:7860 anydisaster-demo
#
#   Training with your data mounted:
#     docker run --rm --gpus all -v "$PWD/data":/workspace/data -v \
#       "$PWD/results":/workspace/results anydisaster \
#       python train.py --config configs/glof/icimod/seg_unet.yaml
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch

# gdal/rasterio system deps; libgl1 for opencv-headless safety
RUN apt-get update && apt-get install -y --no-install-recommends \
        git gdal-bin libgdal-dev libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# dependency layers first for build caching
COPY constraints.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e . -c constraints.txt

# application code
COPY configs ./configs
COPY scripts ./scripts
COPY tools ./tools
COPY tests ./tests
COPY train.py test.py ./

# optional extras (export extras: onnx/onnxscript/onnxruntime)
# RUN pip install --no-cache-dir ".[export]"

CMD ["python", "train.py", "--help"]

# ---------------------------------------------------------------------------
FROM base AS demo
COPY gradio_demo ./gradio_demo
RUN pip install --no-cache-dir "gradio>=4.0"
# Gradio binds to 127.0.0.1 by default, which is unreachable through
# `docker run -p 7860:7860`; listen on all interfaces inside the container.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
EXPOSE 7860
CMD ["python", "gradio_demo/app.py"]
