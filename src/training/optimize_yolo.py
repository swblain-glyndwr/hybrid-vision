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


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(weights: Path, output: Path, prune_pct: float = 0.2) -> None:
    """Load YOLO weights, prune, quantize and save."""
    yolo = YOLO(str(weights))
    core = yolo.model  # nn.Module
    apply_global_pruning(core, amount=prune_pct)
    core = apply_dynamic_quantization(core)
    yolo.model = core
    yolo.save(str(output))
    print(f"Saved optimized model to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prune and quantize a YOLO model")
    p.add_argument("weights", type=Path, help="Path to input YOLO .pt file")
    p.add_argument("output", type=Path, help="Output path for optimized model")
    p.add_argument("--prune", type=float, default=0.2, help="Fraction of weights to prune")
    args = p.parse_args()
    main(args.weights, args.output, prune_pct=args.prune)
