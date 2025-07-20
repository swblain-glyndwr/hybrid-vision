FROM python:3.11-slim-bookworm

# -- system libs + build tools -------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        build-essential ca-certificates wget unzip && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements-edge.txt .
RUN pip install --no-cache-dir -r requirements-edge.txt

COPY src /app/src
ENTRYPOINT ["python", "-m", "src.edge.main_edge"]
