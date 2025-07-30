"""Utility to prune and quantize a YOLOv8 model for edge deployment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from torch.nn.utils import prune
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Pruning helpers
# ---------------------------------------------------------------------------

def _gather_conv_layers(model: torch.nn.Module) -> Iterable[tuple[torch.nn.Module, str]]:
    """Yield (module, parameter_name) tuples for Conv2d layers."""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            yield (module, "weight")


def apply_global_pruning(model: torch.nn.Module, amount: float = 0.2) -> None:
    """Globally prune a fraction of convolution weights."""
    params = list(_gather_conv_layers(model))
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, _ in params:
        prune.remove(mod, "weight")


# ---------------------------------------------------------------------------
# Quantization helper
# ---------------------------------------------------------------------------

def apply_dynamic_quantization(model: torch.nn.Module) -> torch.nn.Module:
    """Apply dynamic quantization to Linear layers."""
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def apply_static_quantization(model: torch.nn.Module) -> torch.nn.Module:
    """Apply post-training static quantization to the entire model."""
    model.eval()
    model.cpu()
    if hasattr(model, "fuse"):
        try:
            model.fuse()
        except Exception:
            pass
    # "qconfig" is an attribute commonly injected into nn.Module instances at
    # runtime, but PyTorch's type hints for ``Module.__setattr__`` only allow
    # ``Tensor`` or ``Module``.  Using ``setattr`` avoids a Pylance type error
    # while keeping the behaviour identical.
    setattr(model, "qconfig", torch.quantization.get_default_qconfig("fbgemm"))
    torch.quantization.prepare(model, inplace=True)
    example = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        model(example)
    torch.quantization.convert(model, inplace=True)
    return model


def apply_qat_quantization(model: torch.nn.Module, steps: int = 100) -> torch.nn.Module:
    """Apply quantization-aware training (QAT) to the entire model."""
    model.train()
    model.cpu()
    if hasattr(model, "fuse"):
        try:
            model.fuse()
        except Exception:
            pass
    # ``qconfig`` assignment via ``setattr`` sidesteps type checking issues in
    # static analysers such as Pylance.
    setattr(model, "qconfig", torch.quantization.get_default_qat_qconfig("fbgemm"))
    torch.quantization.prepare_qat(model, inplace=True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    example = torch.randn(1, 3, 640, 640)
    for _ in range(steps):
        opt.zero_grad()
        out = model(example)
        loss = out.mean()
        loss.backward()
        opt.step()
    model.eval()
    torch.quantization.convert(model, inplace=True)
    return model


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(weights: Path, output: Path, *, prune_pct: float = 0.2,
         quantization: str = "dynamic", steps: int = 100) -> None:
    """Load YOLO weights, prune, quantize and save."""
    yolo = YOLO(str(weights))
    core = yolo.model  # nn.Module
    apply_global_pruning(core, amount=prune_pct)

    if quantization == "dynamic":
        core = apply_dynamic_quantization(core)
    elif quantization == "static":
        core = apply_static_quantization(core)
    elif quantization == "qat":
        core = apply_qat_quantization(core, steps=steps)
    else:
        raise ValueError(f"Unknown quantization method: {quantization}")

    yolo.model = core
    yolo.save(str(output))
    print(f"Saved optimized model to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prune and quantize a YOLO model")
    p.add_argument("weights", type=Path, help="Path to input YOLO .pt file")
    p.add_argument("output", type=Path, help="Output path for optimized model")
    p.add_argument("--prune", type=float, default=0.2, help="Fraction of weights to prune")
    p.add_argument("--quant", choices=["dynamic", "static", "qat"], default="dynamic",
                   help="Quantization method to apply")
    p.add_argument("--steps", type=int, default=100,
                   help="Calibration (static) or training (qat) steps")
    args = p.parse_args()
    main(args.weights, args.output, prune_pct=args.prune,
         quantization=args.quant, steps=args.steps)
