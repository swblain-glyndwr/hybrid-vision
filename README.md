# Hybrid Vision Pipeline

This repository explores a split-compute approach for object detection and segmentation. A lightweight “edge” container runs YOLOv8-n-seg on CPU and sends compressed features to a heavier “cloud” container with GPU access. The overall goal is to measure accuracy, bandwidth, and latency trade-offs while testing a reinforcement-learning controller that decides when to offload processing.
Pipeline Overview

    Edge detection - src/edge/detector.py

        Executes YOLOv8-n-seg using only CPU resources.

        Taps the neck feature map before the segmentation head.

        Compresses that tensor with the codec in src/common/codec.py (baseline zlib or the Adaptive Flow Encoder).

        Packages metadata and compressed bytes into a single MessagePack blob.

    Cloud segmentation - src/cloud/segmenter.py

        Receives the blob and decompresses the feature tensor.

        Reconstructs masks using a light Gen2Seg head.

        Returns masks, timing metrics and the original header.

    Adaptive split controller - src/common/offload_policy.py

        Implements threshold and RL-based policies that decide per frame whether to keep computation local or offload.

Traffic control scripts under tc/ emulate 5G, 4G and 3G links so the entire system can be evaluated on a single machine.
Research Goals

    Model and codec sizing - Determine how small the edge model and how aggressive the feature compression can be while maintaining target mAP on COCO-val.

    Codec efficiency - Measure latency and energy cost of the flow-based codec versus the bandwidth saved.

    Adaptive split - Evaluate the RL controller that chooses between local processing and offloading based on bandwidth and queue depth.

The experiments run inside Docker containers via docker-compose.yml. The edge container can be throttled to Raspberry Pi-class resources (for example an 8 GB Pi 5). If time permits, the same container can run on a real Pi to validate the concept on physical hardware.
Development

Install the Python dependencies from requirements-dev.txt and run tests with pytest:

pip install -r requirements-dev.txt
pytest

Both the edge and cloud containers have their own requirements-*.txt files and Dockerfiles under docker/.

Model Pruning & Quantization
----------------------------

You can prune and quantize the YOLO weights for edge deployment with the helper
script under ``src/training``.  After installing the dev requirements, run:

```
python -m training.optimize_yolo yolov8n-seg.pt pruned_quantized.pt --prune 0.2
```

This loads the given weights, globally prunes a fraction of convolution weights
and applies dynamic quantization before saving the optimized model.

Quick CLI
---------

The ``hv.py`` script in the repository root exposes handy commands so you don't
have to remember the full module paths. A few examples:

```
# Run the edge detector on one image
python hv.py edge path/to/image.jpg

# Run the complete edge → cloud pipeline
python hv.py cloud path/to/image.jpg

# Prune and quantize YOLO weights
python hv.py optimize yolov8n-seg.pt pruned.pt --prune 0.3

# Launch the docker containers
python hv.py compose-up
```

Run ``python hv.py -h`` to see all available subcommands.
