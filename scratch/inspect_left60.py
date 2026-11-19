import sys
sys.path.insert(0, ".")
from pathlib import Path
import cv2
import numpy as np

img = cv2.imread("storage/generated/P01/original/left_60.png")
mask = cv2.imread("storage/deformations/7/P01/broad/left_60_mask.png", cv2.IMREAD_GRAYSCALE)
deformed = cv2.imread("storage/deformations/7/P01/broad/left_60_broad.png")

print("left_60 img shape:", img.shape)
print("left_60 mask non-zero count:", np.count_nonzero(mask), f"({np.count_nonzero(mask) / mask.size * 100:.2f}%)")

ys, xs = np.where(mask > 0)
print(f"Mask bounding box: y=[{ys.min()}, {ys.max()}], x=[{xs.min()}, {xs.max()}]")
print(f"Nose width: {xs.max() - xs.min()}, height: {ys.max() - ys.min()}")

diff = cv2.absdiff(img, deformed)
diff_inside = diff[mask > 0]
print(f"Max diff: {np.max(diff)}, mean diff: {np.mean(diff):.4f}")
print(f"Max diff inside mask: {np.max(diff_inside)}, mean diff inside mask: {np.mean(diff_inside):.4f}")
print(f"Changed pixels: {np.count_nonzero(np.any(diff > 0, axis=-1))} / {diff.shape[0]*diff.shape[1]}")

