FROM python:3.11-slim-bookworm

# system libs + build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        build-essential ca-certificates wget unzip && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements-edge.txt .
RUN pip install --no-cache-dir -r requirements-edge.txt

COPY src /app/src
COPY tc/ /app/tc/  

# Download YOLO weights once at build time
RUN mkdir -p /app/weights && \
    wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt \
         -O /app/weights/yolov8n-seg.pt

ENV YOLO_CONFIG_DIR=/tmp

ENV PYTHONPATH=/app/src:/app
ENTRYPOINT ["python", "-m", "src.edge.detector"]
