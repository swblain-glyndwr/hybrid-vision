"""
Edge-side detector for the hybrid vision pipeline.

Responsibilities:

1. Run YOLOv8-n-seg (CPU) on a single image.
2. Tap an intermediate feature map before the segmentation head.
3. Compress that tensor with common.codec.encode().
4. Package metadata + compressed bytes into one MessagePack-framed blob.

Author: hybrid-vision project (2025-07-20)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import msgpack
import numpy as np
import torch
from ultralytics import YOLO

from common import codec   # local module, baseline zlib (will swap later)


# Configuration constants                                                     #


_MODEL_WEIGHTS = "yolov8n-seg.pt"   # this should already be in edge image
_INPUT_SIZE    = 640               # longest side after resize (square letterbox)
_CONF_THRES    = 0.25
_IOU_THRES     = 0.45
_FEATURE_NODE  = -3                # layer index to tap (neck)
_DEVICE        = torch.device("cpu")


# Utility - Feature tap hook


class _FeatureHook:
    """Registers forward hook on a YOLO layer; stores the last output tensor."""
    def __init__(self):
        self.tensor: torch.Tensor | None = None

    def __call__(self, _module, _in, out: torch.Tensor):
        # Keep a **detached CPU copy** so .numpy() later is zero-copy
        self.tensor = out.detach().to("cpu")


# Public class


class EdgeDetector:
    """
    Example
    -------
    >>> det = EdgeDetector()
    >>> blob = det.run(Path('datasets/coco/val2017/000000000139.jpg'))
    >>> print(f'bytes: {len(blob):,}')
    """
    def __init__(self) -> None:
        self.model = YOLO(_MODEL_WEIGHTS, task="segment")     # auto-download OK
        self.model.fuse()                                     # conv-bn fusion
        self.model.to(_DEVICE)

        # Register hook
        self._hook = _FeatureHook()
        target_layer = self.model.model[_FEATURE_NODE]
        target_layer.register_forward_hook(self._hook)

        # YOLO names for downstream display 
        self.names = self.model.names

    # low-level helpers

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        """Read image as RGB numpy array of shape (H,W,3)."""
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # main API

    def run(self, img_path: Path) -> bytes:
        """
        Run inference on one frame and return a self-contained blob suitable
        for the cloud container. Raises on any failure.
        """
        rgb = self._load_rgb(img_path)

        # forward pass
        t0 = time.perf_counter()
        res = self.model.predict(
            rgb,
            imgsz=_INPUT_SIZE,
            conf=_CONF_THRES,
            iou=_IOU_THRES,
            device=_DEVICE,
            verbose=False,
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1e3

        # feature capture
        feat = self._hook.tensor
        if feat is None:
            raise RuntimeError("Feature hook did not fire.")
        # feat shape: (C,H,W)  -- keep on CPU

        # compression
        t1 = time.perf_counter()
        feat_bytes = codec.encode(feat)             # !!! swap here later !!!
        enc_ms = (time.perf_counter() - t1) * 1e3

        # build header
        meta: Dict[str, Any] = {
            "img_path": str(img_path),
            "shape": tuple(feat.shape),            # (C,H,W)
            "dtype": str(feat.numpy().dtype),      # 'float32'
            "bbox_xyxy": res.boxes.xyxy.cpu().numpy(),    # (N,4)
            "confidence": res.boxes.conf.cpu().numpy(),   # (N,)
            "infer_ms": infer_ms,
            "enc_ms": enc_ms,
        }

        header = msgpack.packb(meta, use_bin_type=True)
        # Prepend 4-byte big-endian header length for easy split on cloud side
        blob = len(header).to_bytes(4, "big") + header + feat_bytes
        return blob


# CLI helper (to help running inside the container manually)


if __name__ == "__main__":    # only executes when `python -m edge.detector`
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    det = EdgeDetector()
    out_blob = det.run(args.image)
    hdr_len = int.from_bytes(out_blob[:4], "big")
    hdr = msgpack.unpackb(out_blob[4:4+hdr_len], strict_map_key=False)
    print(json.dumps({k: (v if isinstance(v, (int, float, str)) else str(type(v)))
                      for k, v in hdr.items()},
                     indent=2))
    print(f"\nTotal bytes on wire: {len(out_blob):,}")
