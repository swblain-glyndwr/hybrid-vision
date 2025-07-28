#!/usr/bin/env python3
"""Simple helper CLI for common hybrid-vision tasks."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import json
import msgpack


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------

def run_edge(image: Path) -> None:
    """Run the edge detector on ``image`` and print the header."""
    from src.edge.detector import EdgeDetector
    det = EdgeDetector()
    blob = det.run(image)
    hdr_len = int.from_bytes(blob[:4], "big")
    hdr = msgpack.unpackb(blob[4:4+hdr_len], strict_map_key=False)
    print(json.dumps(hdr, indent=2))
    print(f"\nTotal bytes on wire: {len(blob):,}")


def run_cloud(image: Path) -> None:
    """Run the full edge → cloud pipeline on ``image`` and show result."""
    from src.edge.detector import EdgeDetector
    from src.cloud.segmenter import CloudSegmenter

    det = EdgeDetector()
    blob = det.run(image)

    seg = CloudSegmenter()
    out = seg.run(blob)
    meta = out["meta"]
    print(json.dumps({k: meta[k] for k in ("infer_ms", "enc_ms")}, indent=2))
    print(f"Masks tensor: {out['masks'].shape}  decode+infer ms: {out['dec_ms']:.1f}")


def optimize(weights: Path, output: Path, prune: float) -> None:
    from src.training.optimize_yolo import main as opt_main
    opt_main(weights, output, prune_pct=prune)


def run_experiment(args: list[str]) -> None:
    cmd = ["python", "-m", "experiments.experiment_runner", *args]
    subprocess.run(cmd, check=True)


def compose_up() -> None:
    subprocess.run(["docker", "compose", "up", "-d"], check=True)


def compose_down() -> None:
    subprocess.run(["docker", "compose", "down"], check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Hybrid Vision helper")
sub = parser.add_subparsers(dest="cmd", required=True)

p_edge = sub.add_parser("edge", help="run edge detector on an image")
p_edge.add_argument("image", type=Path)

p_cloud = sub.add_parser("cloud", help="run edge+cloud on an image")
p_cloud.add_argument("image", type=Path)

p_opt = sub.add_parser("optimize", help="prune and quantize YOLO weights")
p_opt.add_argument("weights", type=Path)
p_opt.add_argument("output", type=Path)
p_opt.add_argument("--prune", type=float, default=0.2)

p_exp = sub.add_parser("experiment", help="run experiment runner")
p_exp.add_argument("args", nargs=argparse.REMAINDER)

sub.add_parser("compose-up", help="docker compose up -d")
sub.add_parser("compose-down", help="docker compose down")

args = parser.parse_args()

if args.cmd == "edge":
    run_edge(args.image)
elif args.cmd == "cloud":
    run_cloud(args.image)
elif args.cmd == "optimize":
    optimize(args.weights, args.output, args.prune)
elif args.cmd == "experiment":
    run_experiment(args.args)
elif args.cmd == "compose-up":
    compose_up()
elif args.cmd == "compose-down":
    compose_down()
