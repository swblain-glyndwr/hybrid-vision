FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        && rm -rf /var/lib/apt/lists/*

# add system packages needed for pip/torch/opencv builds
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         git \
#         build-essential \
#         ffmpeg \
#         libgl1 && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-edge.txt .
RUN pip install --no-cache-dir -r requirements-edge.txt

COPY src /app/src
ENTRYPOINT ["python", "-m", "src.edge.main_edge"]
