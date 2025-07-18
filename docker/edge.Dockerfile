FROM python:3.11-slim

WORKDIR /app
COPY requirements-edge.txt .
RUN pip install --no-cache-dir -r requirements-edge.txt

COPY src /app/src
ENTRYPOINT ["python", "-m", "src.edge.main_edge"]
