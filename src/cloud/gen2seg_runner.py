from __future__ import annotations

"""Wrapper around the Gen2Seg Stable Diffusion pipeline for quick inference."""

from pathlib import Path
import torch
from PIL import Image

try:
    from gen2seg_sd_pipeline import gen2segSDPipeline
except Exception as e:  # pragma: no cover - optional dependency
    gen2segSDPipeline = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def run(image: Path, model: str = "reachomk/gen2seg-sd") -> torch.Tensor:
    """Run Gen2Seg SD model on ``image`` and return the mask tensor."""
    if gen2segSDPipeline is None:
        raise RuntimeError(f"gen2seg package is unavailable: {_IMPORT_ERROR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = gen2segSDPipeline.from_pretrained(model, use_safetensors=True).to(device)
    img = Image.open(image).convert("RGB")

    with torch.no_grad():
        out = pipe(img, output_type="pt")

    pred = out.prediction
    return pred.squeeze(0) if torch.is_tensor(pred) else torch.from_numpy(pred).squeeze(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Gen2Seg SD inference on an image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=str, default="reachomk/gen2seg-sd")
    args = parser.parse_args()

    mask = run(args.image, args.model)
    print(f"Mask shape: {tuple(mask.shape)}")
