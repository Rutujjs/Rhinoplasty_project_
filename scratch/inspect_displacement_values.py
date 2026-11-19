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

print(f"Image: {w}x{h}, Mask pixels: {np.count_nonzero(mask)}")
ys, xs = np.where(mask > 0)
nose_w = xs.max() - xs.min() + 1
nose_h = ys.max() - ys.min() + 1
print(f"Nose width: {nose_w}, height: {nose_h}")

from deformation.algorithms import deviated_nose_deformation, frontal_saddle_nose, frontal_broad_nose, frontal_hump_nose

# Test deviated
dev_res = deviated_nose_deformation(img, mask, strength=0.35, direction="right")
diff_dev = cv2.absdiff(img, dev_res)
print(f"Deviated (str=0.35): max diff={diff_dev.max()}, mean diff in mask={diff_dev[mask>0].mean():.2f}")

# Test saddle
sad_res = frontal_saddle_nose(img, mask, strength=0.65)
diff_sad = cv2.absdiff(img, sad_res)
print(f"Saddle (str=0.65): max diff={diff_sad.max()}, mean diff in mask={diff_sad[mask>0].mean():.2f}")

# Test broad
brd_res = frontal_broad_nose(img, mask, strength=0.40)
diff_brd = cv2.absdiff(img, brd_res)
print(f"Broad (str=0.40): max diff={diff_brd.max()}, mean diff in mask={diff_brd[mask>0].mean():.2f}")

# Test hump
hmp_res = frontal_hump_nose(img, mask, strength=0.40)
diff_hmp = cv2.absdiff(img, hmp_res)
print(f"Hump (str=0.40): max diff={diff_hmp.max()}, mean diff in mask={diff_hmp[mask>0].mean():.2f}")
