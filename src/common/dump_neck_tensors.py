# scripts/dump_neck_tensors.py
import argparse, torch, tqdm
from pathlib import Path
from edge.detector import EdgeDetector

def main(img_dir: Path, out_file: Path, max_imgs: int):
    edge = EdgeDetector()
    feats = []
    for idx, jpg in enumerate(tqdm.tqdm(sorted(img_dir.glob("*.jpg")))):
        if idx == max_imgs:
            break
        neck = edge.extract_neck(jpg)          # tiny change in EdgeDetector (see below)
        feats.append(neck.view(-1))               # flatten to 1-D
    stack = torch.stack(feats)                    # [N, D]
    torch.save(stack, out_file)
    print(f"Wrote {out_file} ({stack.shape[0]} tensors).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs",  default="datasets/coco/val2017")
    ap.add_argument("--out",   default="results/neck_stack.pt")
    ap.add_argument("--count", type=int, default=2000,
                    help="number of images to sample")
    args = ap.parse_args()
    main(Path(args.imgs), Path(args.out), args.count)
