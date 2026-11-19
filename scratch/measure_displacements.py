import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from pathlib import Path

p01_dir = Path("storage/generated/P01/original")
img = cv2.imread(str(p01_dir / "frontal.png"))
h, w = img.shape[:2]

from deformation.bisenet_inference import bisenet_segmentor
mask, meta = bisenet_segmentor.predict_nose_mask(img, view_name="frontal")

ys, xs = np.where(mask > 0)
y_top, y_bottom = int(ys.min()), int(ys.max())
x_left, x_right = int(xs.min()), int(xs.max())
nose_h = y_bottom - y_top + 1
nose_w = x_right - x_left + 1

# Check deviated formula
strength = 0.35
max_displacement = strength * nose_w * 0.30
print(f"Deviated formula max_displacement: {max_displacement:.2f} pixels (nose_w={nose_w})")

# Check saddle formula in deformation/algorithms.py
# Let's inspect saddle formula
print(f"For saddle (str=0.65): ...")

