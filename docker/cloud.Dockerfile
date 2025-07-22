FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# add system packages needed for pip/torch/opencv builds
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ffmpeg \
        wget \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*


# clone Gen2Seg and add in a pythonpath that we'll define later
# this is needed for the cloud container to find the gen2seg package
RUN git clone --depth 1 https://github.com/UCDvision/gen2seg.git /opt/gen2seg
ENV PYTHONPATH="/opt/gen2seg:${PYTHONPATH}"

RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m pip install --upgrade pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
    ln -s /usr/bin/python3.11 /usr/local/bin/python

WORKDIR /app
COPY requirements-cloud.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements-cloud.txt

COPY src /app/src
COPY tc/ /app/tc/  


# Download YOLO weights once at build time
RUN mkdir -p /app/weights && \
    wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt \
         -O /app/weights/yolov8n-seg.pt

ENV YOLO_CONFIG_DIR=/tmp

ENV PYTHONPATH=/app/src:/app
ENTRYPOINT ["python3.11", "-m", "src.cloud.segmenter"]