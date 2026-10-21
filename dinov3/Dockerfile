# DINOv3 feature-extraction worker for RunPod Serverless. Weights come from the network volume at run time.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ARG DINOV3_SHA=6876159a11b4df116f30f667f8c9888617df0751
ENV PYTHONPATH=/opt/dinov3 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir runpod pillow numpy && \
    python -c "import urllib.request,tarfile,io; tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen('https://github.com/facebookresearch/dinov3/archive/${DINOV3_SHA}.tar.gz').read())).extractall('/opt')" && \
    mv /opt/dinov3-${DINOV3_SHA} /opt/dinov3 && \
    python -c "import runpod, torch; from dinov3.hub import backbones; m = backbones.dinov3_vit7b16(pretrained=False, device='meta'); print('imports ok', torch.__version__)"
COPY dinov3_handler.py /opt/dinov3_handler.py
CMD ["python", "-u", "/opt/dinov3_handler.py"]
