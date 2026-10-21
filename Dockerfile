# SAM 3.1 RunPod Serverless worker.
# Base: community image with Meta's sam3 code (SAM 3.1 support) at /app/sam3 and a Python 3.12 venv (torch 2.10, cu128).
# Adds: RunPod SDK, setuptools<80 (pkg_resources), any module the sam3 import still misses, and the handler.
# No model weights inside (Meta's SAM licence gates them): the worker downloads sam3.1_multiplex.pt at cold start.
FROM muhammadshahmirzaman/sam3.1:slim

ENV PYTHONPATH=/app/sam3 \
    PYTHONUNBUFFERED=1 \
    SAM31_CKPT=/checkpoints/sam3.1_multiplex.pt

COPY install_deps.py /opt/install_deps.py
RUN python -m pip --version >/dev/null 2>&1 || python -m ensurepip; \
    python -m pip install --no-cache-dir runpod "setuptools<80" && \
    python /opt/install_deps.py && \
    python -c "import runpod, pkg_resources; from sam3.model_builder import build_sam3_predictor; print('imports ok')"

COPY sam31_handler.py /opt/sam31_handler.py
WORKDIR /app/sam3
CMD ["python", "-u", "/opt/sam31_handler.py"]
