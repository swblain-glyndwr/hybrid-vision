import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge import detector as det_mod
from cloud.segmenter import CloudSegmenter


def test_full_pipeline(tmp_path):
    # ensure YOLO weights download to a writable location
    det_mod._MODEL_WEIGHTS = str(tmp_path / "yolov8n-seg.pt")

    # create dummy input image
    img_path = tmp_path / "test.jpg"
    img = Image.fromarray(np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8))
    img.save(img_path)

    det = det_mod.EdgeDetector()
    blob = det.run(img_path)

    seg = CloudSegmenter()
    out = seg.run(blob)

    # masks tensor has shape (1, H, W)
    masks = out["masks"]
    assert masks.ndim == 3
    assert masks.shape[0] == 1

