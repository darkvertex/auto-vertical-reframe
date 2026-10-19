# Runpod Serverless worker. Cog builds from cog.yaml and does not read this file.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/src/src \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && printf '%s\n' \
        'deb http://archive.ubuntu.com/ubuntu jammy universe' \
        'deb http://archive.ubuntu.com/ubuntu jammy-updates universe' \
        > /etc/apt/sources.list.d/universe.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
    && python3.11 -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

COPY replicate-requirements.txt /tmp/replicate-requirements.txt
RUN pip install --no-cache-dir -r /tmp/replicate-requirements.txt \
    && pip install --no-cache-dir "runpod>=1.7.13" "boto3>=1.35"

# Detection weights stay cached across code changes. The segmentation
# checkpoint is copied with the repo. Mirrors the curl in cog.yaml.
RUN mkdir -p /src \
    && curl -fsSL -o /src/yolo11n.pt \
        https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt \
    && test "$(stat -c%s /src/yolo11n.pt)" -gt 5000000 \
    && curl -fsSL -o /src/yoloe-11s-seg.pt \
        https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg.pt \
    && test "$(stat -c%s /src/yoloe-11s-seg.pt)" -gt 5000000

WORKDIR /src
COPY . /src

RUN python -c 'from ultralytics import settings; settings.update({"sync": False}); from ultralytics.utils.checks import check_font; check_font("Arial.ttf")'

CMD ["python", "-u", "/src/handler.py"]
