from pathlib import Path
import cv2
import numpy as np

orig_path = Path("storage/generated/P01/original/frontal.png")
mask_path = Path("storage/deformations/7/P01/broad/frontal_mask.png")
def_path = Path("storage/deformations/7/P01/broad/frontal_broad.png")

orig = cv2.imread(str(orig_path))
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
deformed = cv2.imread(str(def_path))

print(f"Orig shape: {orig.shape}")
print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
print(f"Mask non-zero count: {np.count_nonzero(mask)}, percentage of image: {np.count_nonzero(mask) / mask.size * 100:.2f}%")

ys, xs = np.where(mask > 0)
print(f"Mask bounding box: y=[{ys.min()}, {ys.max()}], x=[{xs.min()}, {xs.max()}]")
print(f"Mask centroid: y={np.mean(ys):.1f}, x={np.mean(xs):.1f}")
print(f"Nose width: {xs.max() - xs.min()}, height: {ys.max() - ys.min()}")

# Now inspect difference inside vs outside mask
diff = cv2.absdiff(orig, deformed)
diff_inside = diff[mask > 0]
diff_outside = diff[mask == 0]

print(f"Inside mask: max diff = {np.max(diff_inside)}, mean diff = {np.mean(diff_inside):.4f}, changed pixels = {np.count_nonzero(np.any(diff_inside > 0, axis=-1))} / {len(diff_inside)} ({np.count_nonzero(np.any(diff_inside > 0, axis=-1)) / len(diff_inside) * 100:.2f}%)")
print(f"Outside mask: max diff = {np.max(diff_outside)}, mean diff = {np.mean(diff_outside):.4f}, changed pixels = {np.count_nonzero(np.any(diff_outside > 0, axis=-1))} / {len(diff_outside)} ({np.count_nonzero(np.any(diff_outside > 0, axis=-1)) / len(diff_outside) * 100:.2f}%)")

