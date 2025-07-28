"""Codec utilities used by both the edge and cloud containers."""

from __future__ import annotations

import os
import zlib
from pathlib import Path
from typing import Tuple, Sequence

import torch
import torch.nn as nn

__all__ = [
    "AdaptiveFlowEncoder",
    "_ckpt",
    "set_backend",
    "encode",
    "decode",
]


# ---------------------------------------------------------------------------
# Codec backends
# ---------------------------------------------------------------------------

_BACKEND = os.environ.get("HYBRID_CODEC", "zlib").lower()

# default checkpoint path; user is expected to mount this when using AFE
_ckpt = Path("/app/weights/codec.pt")

_afe_model: "AdaptiveFlowEncoder | None" = None


class AdaptiveFlowEncoder(nn.Module):
    """Minimal placeholder autoencoder used for the AFE backend."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.enc = nn.Conv2d(channels, channels, 1)
        self.dec = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.dec(self.enc(x))

    def criterion(self, x: torch.Tensor, mu: float = 1e-3) -> Tuple[torch.Tensor, float]:
        z = self.enc(x)
        rec = self.dec(z)
        rec_loss = torch.mean((rec - x) ** 2)
        reg_loss = torch.mean(z ** 2)
        loss = rec_loss + mu * reg_loss
        return loss, rec_loss.item()


def _load_afe(channels: int) -> AdaptiveFlowEncoder:
    global _afe_model
    if _afe_model is None:
        _afe_model = AdaptiveFlowEncoder(channels)
        if _ckpt.exists():
            _afe_model.load_state_dict(torch.load(_ckpt, map_location="cpu"))
        _afe_model.eval()
    return _afe_model


def set_backend(name: str) -> None:
    """Select the codec backend globally ("zlib" or "afe")."""

    global _BACKEND
    name = name.lower()
    if name not in {"zlib", "afe"}:
        raise ValueError("codec must be 'zlib' or 'afe'")
    _BACKEND = name


def _encode_zlib(tensor: torch.Tensor) -> bytes:
    return zlib.compress(tensor.numpy().tobytes(), level=3)


def _dtype_from_any(d: str | torch.dtype) -> torch.dtype:
    if isinstance(d, torch.dtype):
        return d
    return getattr(torch, d)


def _decode_zlib(blob: bytes, dtype: str | torch.dtype, shape: Sequence[int]) -> torch.Tensor:
    arr = zlib.decompress(blob)
    return torch.frombuffer(arr, dtype=_dtype_from_any(dtype)).view(*shape)


def _encode_afe(tensor: torch.Tensor) -> bytes:
    model = _load_afe(tensor.shape[0])
    with torch.no_grad():
        lat = model.enc(tensor.unsqueeze(0)).squeeze(0).to(torch.float16)
    return zlib.compress(lat.numpy().tobytes(), level=3)


def _decode_afe(blob: bytes, dtype: str | torch.dtype, shape: Sequence[int]) -> torch.Tensor:
    model = _load_afe(shape[0])
    arr = zlib.decompress(blob)
    lat = torch.frombuffer(arr, dtype=torch.float16).view(*shape)
    with torch.no_grad():
        out = model.dec(lat.unsqueeze(0)).squeeze(0)
    return out.to(_dtype_from_any(dtype))


def encode(tensor: torch.Tensor) -> bytes:
    """Compress ``tensor`` using the selected backend."""

    if _BACKEND == "afe":
        return _encode_afe(tensor)
    return _encode_zlib(tensor)


def decode(blob: bytes, dtype: str | torch.dtype, shape: Sequence[int]) -> torch.Tensor:
    """Decompress ``blob`` using the selected backend."""

    if _BACKEND == "afe":
        return _decode_afe(blob, dtype, shape)
    return _decode_zlib(blob, dtype, shape)

