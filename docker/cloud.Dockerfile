FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m pip install --upgrade pip

WORKDIR /app
COPY requirements-cloud.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements-cloud.txt

COPY src /app/src
ENTRYPOINT ["python3.11", "-m", "src.cloud.main_cloud"]