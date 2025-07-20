"""
Hybrid-Vision Experiment Harness
================================

Runs a batch of frames through the edge detector -> codec -> cloud segmenter
pipeline, measures latency, bandwidth and (optionally) mAP.

Usage (host or WSL shell)
-------------------------
python -m experiments.experiment_runner \
        --dataset  datasets/coco/val2017 \
        --frames   200 \
        --profile  5G \
        --out      results/run1.jsonl

If you want to run *inside* a container, make sure the dataset is mounted at
the same path given by --dataset.

Author: hybrid-vision project — 2025-07-20
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from tqdm import tqdm

# Local modules
from edge.detector import EdgeDetector
from cloud.segmenter import CloudSegmenter

# uncomment when Gen2Seg head is trained
# from torchmetrics.detection.mean_ap import MeanAveragePrecision
# import torch


# Traffic-control helper


def _apply_tc(profile: str) -> None:
    """
    Call tc/apply_tc.py if it exists; otherwise print a warning.
    """
    helper = Path("tc/apply_tc.py")
    if not helper.exists():
        print(f"[WARN] TC helper not found → skipping shaping (profile='{profile}')")
        return
    import subprocess, sys
    try:
        subprocess.run(
            [sys.executable, str(helper), "--profile", profile],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[WARN] TC helper failed: {e.stderr.strip() or e}")
    else:
        print(f"[INFO] Applied TC profile '{profile}'")


# Metrics utilities


class OnlineStats:
    """Simple Welford mean/variance + quantiles."""
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.values: List[float] = []

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
        self.values.append(x)

    def result(self) -> Dict[str, float]:
        if self.n == 0:
            return {}
        p95 = float(np.percentile(self.values, 95))
        std = (self.M2 / (self.n - 1)) ** 0.5 if self.n > 1 else 0.0
        return {"mean": self.mean, "p95": p95, "std": std}


# Core experiment runner


def run_experiment(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset).expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    # Apply traffic-control shaping (host loopback)
    _apply_tc(args.profile)

    # Instantiate edge & cloud
    edge = EdgeDetector()
    cloud = CloudSegmenter()

    # mAP metric
    # map_metric = MeanAveragePrecision().to("cpu")

    # Prepare output file
    results_path = Path(args.out)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    fout = results_path.open("w")

    # Iterate frames
    lat_stats = defaultdict(OnlineStats)   # encode_ms, decode_ms, total_ms
    size_stats = OnlineStats()

    images = sorted(dataset_root.glob("*.jpg"))
    if args.frames:
        images = images[: args.frames]

    for img_path in tqdm(images, desc="Frames"):
        t0 = time.perf_counter()
        blob = edge.run(img_path)
        enc_time = edge._hook.tensor is not None and edge._hook.tensor  # not used
        # on-wire size
        size_stats.add(len(blob))

        # cloud side
        out = cloud.run(blob)
        t_total = (time.perf_counter() - t0) * 1e3

        # accumulate stats
        lat_stats["decode_ms"].add(out["dec_ms"])
        lat_stats["total_ms"].add(t_total)
        lat_stats["encode_ms"].add(out["meta"]["enc_ms"])

        # update mAP object
        # preds = dict(
        #     masks=torch.sigmoid(out["logits"]) > 0.5,
        #     scores=torch.tensor(out["meta"]["confidence"]),
        #     labels=torch.zeros_like(out["meta"]["confidence"], dtype=torch.long),
        # )
        # target = ...  # load from COCO instances JSON
        # map_metric.update([preds], [target])

        # Write per-frame JSONL
        frame_record = {
            "img": str(img_path),
            "bytes": len(blob),
            "infer_ms": out["meta"]["infer_ms"],
            "enc_ms": out["meta"]["enc_ms"],
            "dec_ms": out["dec_ms"],
            "total_ms": t_total,
            "timestamp": time.time(),
        }
        fout.write(json.dumps(frame_record) + "\n")

    fout.close()

    # Print summary
    print("\n=== Summary ===")
    for k, stat in lat_stats.items():
        res = stat.result()
        print(f"{k:<10}  mean: {res['mean']:.2f} ms   p95: {res['p95']:.2f} ms")
    size_res = size_stats.result()
    print(f"bytes      mean: {size_res['mean']:.0f} B   p95: {size_res['p95']:.0f} B")

    # Final mAP
    # map_res = map_metric.compute()
    # print(f"mAP50-95: {map_res['map']:.3f}")

    print(f"[✓] Saved per-frame results to {results_path}")


# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid-Vision Experiment Runner")
    p.add_argument(
        "--dataset",
        type=str,
        default="datasets/coco/val2017",
        help="Path to folder of .jpg frames",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Number of frames to process (0 = all)",
    )
    p.add_argument(
        "--profile",
        type=str,
        default="none",
        help="TC profile name (5G, 4G, 3G, none)",
    )
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    p.add_argument(
        "--out",
        type=str,
        default=f"results/run_{dt}.jsonl",
        help="Output JSONL path",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
