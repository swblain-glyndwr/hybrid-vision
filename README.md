# Hybrid Vision Pipeline

This repository explores a split-compute approach for object detection and segmentation. A lightweight edge container runs YOLOv8-n-seg on CPU and sends compressed features to a heavier cloud container with GPU access. The aim is to measure accuracy, bandwidth and latency trade-offs while experimenting with a reinforcement-learning controller that decides when to offload processing.

## Pipeline Overview

### Edge Detection (`src/edge/detector.py`)
- Runs YOLOv8-n-seg using only CPU resources.
- Taps the neck feature map before the segmentation head.
- Compresses that tensor with the codec in `src/common/codec.py` (baseline zlib or the Adaptive Flow Encoder).
- Packages metadata and compressed bytes into a single MessagePack blob.

### Cloud Segmentation (`src/cloud/segmenter.py`)
- Receives the blob and decompresses the feature tensor.
- Reconstructs masks with a lightweight Gen2Seg head.
- Returns masks, timing metrics and the original header.

### Adaptive Split Controller (`src/common/offload_policy.py`)
- Implements threshold and RL-based policies that decide per frame whether to keep computation local or offload.
- Works with traffic control scripts in `tc/` that emulate 5G, 4G and 3G links so the entire system can be evaluated on a single machine.

## Research Goals

- **Model and codec sizing** – determine how small the edge model can be and how aggressively features can be compressed while maintaining target mAP on COCO-val.
- **Codec efficiency** – measure latency and energy cost of the flow-based codec versus the bandwidth saved.
- **Adaptive split** – evaluate the RL controller that chooses between local processing and offloading based on bandwidth and queue depth.

The experiments run inside Docker containers via `docker-compose.yml`. The edge container can be throttled to Raspberry Pi‑class resources (for example an 8 GB Pi 5). If time permits, the same container can run on a real Pi to validate the concept on physical hardware.

## Development

1. Install the Python dependencies from `requirements-dev.txt`.
2. Run `pytest` to execute the unit tests.
3. Both the edge and cloud containers have their own `requirements-*.txt` files and Dockerfiles under `docker/`.

### Model Pruning & Quantization

You can prune and quantize the YOLO weights for edge deployment with the helper
script under `src/training`. The tool supports **dynamic**, **static** and
**quantization-aware training (QAT)** modes:

```bash
python -m training.optimize_yolo yolov8n-seg.pt pruned_quantized.pt --prune 0.2 --quant static
```

By default dynamic quantization is used. Pass `--quant static` for post-training
static quantization or `--quant qat` (optionally with `--steps`) to run a short
QAT loop before converting the model.

### Quick CLI

The `hv.py` script in the repository root exposes handy commands so you don't have to remember full module paths:

```bash
# Run the edge detector on one image
python hv.py edge path/to/image.jpg

# Run the complete edge → cloud pipeline
python hv.py cloud path/to/image.jpg

# Prune and quantize YOLO weights (static quantization example)
python hv.py optimize yolov8n-seg.pt pruned.pt --prune 0.3 --quant static

# Launch the docker containers
python hv.py compose-up
```

Run `python hv.py -h` to see all available subcommands.

