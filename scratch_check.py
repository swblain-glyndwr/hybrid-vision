from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics.data.augment import LetterBox

img_path = Path(r"C:\datasets\coco\val2017\000000000139.jpg")

# RGB → BGR, contiguous
rgb = Image.open(img_path).convert("RGB")
bgr = np.asarray(rgb, dtype=np.uint8)[:, :, ::-1].copy()

print("array info:", type(bgr), bgr.shape, bgr.dtype, bgr.flags['C_CONTIGUOUS'])

# Ultralytics' new API
lb = LetterBox(new_shape=640, stride=32)
out = lb(image=bgr)
print("letterbox OK, output shape:", out.shape)
