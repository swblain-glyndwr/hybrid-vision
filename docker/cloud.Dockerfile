FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# add system packages needed for pip/torch/opencv builds
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ffmpeg \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*


# clone Gen2Seg and add in a pythonpath that we'll define later
# this is needed for the cloud container to find the gen2seg package
RUN git clone --depth 1 https://github.com/UCDvision/gen2seg.git /opt/gen2seg
ENV PYTHONPATH="/opt/gen2seg:${PYTHONPATH}"

RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m pip install --upgrade pip
RUN ln -s /usr/bin/python3.11 /usr/local/bin/python

WORKDIR /app
COPY requirements-cloud.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements-cloud.txt

COPY src /app/src
ENTRYPOINT ["python3.11", "-m", "src.cloud.main_cloud"]