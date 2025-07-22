"""
cloud/segmenter.py
------------------
1. Accept a binary blob produced by edge.detector.EdgeDetector
   (4-byte header length  +  MessagePack header  +  compressed features).
2. Decompress the feature tensor with common.codec.decode().
3. Feed the tensor to a light-weight Gen2Seg head → class-agnostic masks.
4. Return a dictionary with logits, meta header and timing.

Author: hybrid-vision project (2025-07-20)
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import msgpack

from common import codec                       # same stub as edge side


# Gen2Seg – placeholder (tiny FPN)

class Gen2Seg(nn.Module):
    """
    Very small 3-layer head that upsamples features back to 1/4-scale.
    Replace later with your trained Gen2Seg weights.
    """
    def __init__(self, c_in: int, stride: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 1)          # 1-channel class-agnostic mask
        self.up    = nn.Upsample(scale_factor=stride, mode="bilinear",
                                 align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))          # (B,1,H/stride,W/stride)
        x = self.up(x)
        return x.squeeze(1)                       # (B,H,W)


# Segmenter class

class CloudSegmenter:
    """
    Example
    -------
    >>> seg = CloudSegmenter()
    >>> blob = Path('sample_blob.bin').read_bytes()
    >>> out = seg.run(blob)
    >>> print(out['masks'].shape)
    """
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head   = None      # build lazily per C-in

    # helpers
    @staticmethod
    def _split_blob(blob: bytes) -> Tuple[Dict[str, Any], bytes]:
        """Return (meta dict, compressed bytes)."""
        hdr_len = int.from_bytes(blob[:4], "big")
        meta = msgpack.unpackb(blob[4:4+hdr_len], strict_map_key=False)
        comp = blob[4+hdr_len:]
        return meta, comp

    # API
    def run(self, blob: bytes) -> Dict[str, Any]:
        t0 = time.perf_counter()

        meta, comp = self._split_blob(blob)
        feat = codec.decode(comp, dtype=meta["dtype"], shape=meta["shape"])
        C, H, W = feat.shape
        # (re)build head if channel count changed
        if self.head is None or self.head.conv1.in_channels != C:
            self.head = Gen2Seg(c_in=C).to(self.device).eval()
        feat = feat.unsqueeze(0).to(self.device, non_blocking=True)   # (1,C,H,W)

        with torch.no_grad():
            masks = self.head(feat)                                   # (1,H,W)

        infer_ms = (time.perf_counter() - t0) * 1e3
        return {"masks": masks.cpu(), "meta": meta, "dec_ms": infer_ms}


# CLI helper – run inside container

if __name__ == "__main__":
    import argparse, json
    from edge.detector import EdgeDetector          # local import for demo
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path,
                        help="Test image path (will run edge then cloud).")
    args = parser.parse_args()

    # Generate blob using edge detector (CPU in same container for demo)
    det = EdgeDetector()
    blob = det.run(args.image)

    # Decode & segment
    seg = CloudSegmenter()
    out = seg.run(blob)
    print(json.dumps({k: (v if isinstance(v, (int, float, str))
                          else str(type(v)))
                      for k, v in out["meta"].items()},
                     indent=2))
    print(f"Masks tensor: {out['masks'].shape}  decode+infer ms: {out['dec_ms']:.1f}")

