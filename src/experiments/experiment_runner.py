"""
Hybrid-Vision Experiment Harness
================================

Runs a batch of frames through the edge detector -> codec -> cloud segmenter
pipeline, measures latency, bandwidth and (optionally) mAP.

Usage (host or WSL shell)
-------------------------
python -m experiments.experiment_runner \
        --dataset  datasets/coco \
        --frames   200 \
        --profile  5G \
        --out      results/run1.jsonl

If you want to run *inside* a container, make sure the dataset is mounted at
the same path given by --dataset.

Author: hybrid-vision project — 2025-07-20
"""

from __future__ import annotations


import argparse, csv, json, time
from collections import defaultdict
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from pycocotools.coco import COCO

from tc.apply_tc import apply_profile   # your helper

# default dataset root used when --dataset isn't provided
DEFAULT_DATA_ROOT = Path("/data/coco")

# category-id mapping is populated once the dataset path is known
COCO_CAT_ID_TO_IDX = {}


# Helper: map YOLO xyxy numpy → torchmetric dicts

def yolo_to_pred(bboxes, conf, labels):
    """Return list[dict] with 'boxes', 'scores' and 'labels' ."""
    return [{
        "boxes":  torch.as_tensor(bboxes,  dtype=torch.float32),
        "scores": torch.as_tensor(conf,    dtype=torch.float32),
        "labels": torch.as_tensor(labels,  dtype=torch.long),
    }]

def coco_gt(coco: COCO, img_id: int):
    """Return dict for torchmetrics (boxes + labels) for one COCO image id."""
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=False))
    boxes, labels = [], []
    for a in anns:
        x, y, w, h = a["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(COCO_CAT_ID_TO_IDX[a["category_id"]])   # remap!
    return [{
        "boxes":  torch.tensor(boxes,  dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }]


# Main runner

def main(args):
    DATA_ROOT = args.dataset
    val_imgs  = DATA_ROOT / "val2017"
    ann_file  = DATA_ROOT / "annotations" / "instances_val2017.json"

    # Build mapping {coco_id -> 0-79 contiguous}
    coco_cat_ids = sorted(COCO(str(ann_file)).getCatIds())
    global COCO_CAT_ID_TO_IDX
    COCO_CAT_ID_TO_IDX = {cid: i for i, cid in enumerate(coco_cat_ids)}

    # optional traffic-control
    if args.profile:
        apply_profile(args.profile)

    # detectors
    edge  = EdgeDetector()
    cloud = CloudSegmenter()

    # COCO helper & metric
    coco   = COCO(str(ann_file))
    metric = MeanAveragePrecision(
            iou_type="bbox",
            iou_thresholds=[x / 100 for x in range(50, 100, 5)],   # 0.50‥0.95
    )
    stats  = []                     # per-frame CSV rows

    img_files = sorted(val_imgs.glob("*.jpg"))[:args.frames]
    for img_path in tqdm(img_files, desc="frames"):
        t_all0 = time.perf_counter()

        blob = edge.run(img_path)
        out  = cloud.run(blob)

        # collect latency & size
        meta = out["meta"]
        row  = {
            "file":   img_path.name,
            "edge_ms":  meta["infer_ms"],
            "enc_ms":   meta["enc_ms"],
            "blob_kB":  len(blob)/1024,
            "cloud_ms": out["dec_ms"],
            "total_ms": (time.perf_counter()-t_all0)*1e3,
        }
        stats.append(row)

        # accumulate mAP (bbox only for now)
        img_id = int(img_path.stem)
        preds  = yolo_to_pred(meta["bbox_xyxy"], meta["confidence"], meta["labels"])
        gts    = coco_gt(coco, img_id)
        metric.update(preds, gts)

    # results
    mp = metric.compute()
    summary = {
        "mAP50-95": float(mp["map"]),
        "mAP50": float(mp["map_50"]),
        "mean_total_ms": sum(r["total_ms"] for r in stats) / len(stats),
        "mean_blob_kB": sum(r["blob_kB"] for r in stats) / len(stats),
    }

    # CSV dump
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=stats[0].keys())
            w.writeheader(); w.writerows(stats)
        print(f"saved per-frame stats → {args.csv}")

        summary_path = args.csv.parent / f"{args.csv.stem}_summary{args.csv.suffix}"
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary.keys())
            w.writeheader(); w.writerow(summary)
        print(f"saved summary stats → {summary_path}")


# CLI

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATA_ROOT,
                   help="path to COCO dataset root")
    p.add_argument("--frames", type=int, default=50,
                   help="number of COCO-val images (default 50)")
    p.add_argument("--profile", type=str, default=None,
                   help="TC profile name as understood by tc/apply_tc.py")
    p.add_argument("--codec", type=str, choices=["zlib", "afe"], default="zlib",
                   help="codec backend to use: zlib or afe")
    p.add_argument("--csv", type=Path, default=None,
                   help="optional output CSV file")
    args = p.parse_args()

    from common import codec
    codec.set_backend(args.codec)
    from edge.detector import EdgeDetector
    from cloud.segmenter import CloudSegmenter

    main(args)
