"""
common/codec.py
A thin wrapper so we can swap the baseline zlib codec
for the learned Adaptive Flow Encoder later with minimal
changes anywhere else.
"""

from __future__ import annotations
import zlib
import torch
from typing import Tuple

def encode(tensor: torch.Tensor) -> bytes:          # tensor: (C,H,W) on *CPU*
    return zlib.compress(tensor.numpy().tobytes(), level=3)

def decode(blob: bytes,
           dtype: str,
           shape: Tuple[int, int, int]) -> torch.Tensor:     # returns CPU tensor
    arr = zlib.decompress(blob)
    return torch.frombuffer(arr, dtype=getattr(torch, dtype)).view(*shape)
