import sys
sys.path.insert(0, ".")
from pathlib import Path
import cv2
import numpy as np

img = cv2.imread("storage/generated/P01/original/frontal.png")
mask = cv2.imread("storage/deformations/7/P01/broad/frontal_mask.png", cv2.IMREAD_GRAYSCALE)
h, w = img.shape[:2]

# Let's inspect _broad_nose_warp step-by-step
from deformation.algorithms import _broad_nose_warp

# Let's recreate the maps inside _broad_nose_warp
ys, xs = np.where(mask > 0)
y_top, y_bottom = int(ys.min()), int(ys.max())
x_left, x_right = int(xs.min()), int(xs.max())
nose_height = y_bottom - y_top + 1
nose_width = x_right - x_left + 1

centerline = np.full(h, np.nan, dtype=np.float32)
for y in range(y_top, y_bottom + 1):
    row_x = np.flatnonzero(mask[y])
    if row_x.size:
        centerline[y] = (row_x.min() + row_x.max()) / 2.0

valid_rows = np.flatnonzero(np.isfinite(centerline))
centerline = np.interp(np.arange(h), valid_rows, centerline[valid_rows]).astype(np.float32)
centerline = cv2.GaussianBlur(centerline.reshape(-1, 1), (0, 0), sigmaX=max(1.0, nose_height * 0.012)).ravel()

yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
t = np.clip((np.arange(h, dtype=np.float32) - y_top) / max(nose_height - 1, 1), 0.0, 1.0)
lower_center, lower_width = 0.70, 0.25
view_factor = 1.0
vertical_profile = np.exp(-0.5 * ((t - lower_center) / lower_width) ** 2) * (np.clip(np.sin(np.pi * t), 0.0, 1.0) ** 0.7)
vertical_profile *= ((np.arange(h) >= y_top) & (np.arange(h) <= y_bottom))

kernel_size = max(5, int(round(nose_width * 0.12)) | 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
expanded = cv2.dilate(mask, kernel, iterations=1)
support = cv2.GaussianBlur(expanded.astype(np.float32), (0, 0), sigmaX=max(1.0, nose_width * 0.025))
support = np.clip(support, 0.0, 1.0)

horizontal_distance = xx - centerline[:, None]
strength = 0.45
amount = float(strength) * view_factor * vertical_profile[:, None]

# Raw displacement without blending
map_x_raw = (centerline[:, None] + horizontal_distance / (1.0 + amount)).astype(np.float32)
disp_x_raw = map_x_raw - xx

# With support blending on map_x
map_x_blended = xx + (map_x_raw - xx) * support
disp_x_blended = map_x_blended - xx

print(f"Nose width: {nose_width}px, height: {nose_height}px")
print(f"Max raw displacement: {np.max(np.abs(disp_x_raw)):.2f}px")
print(f"Max blended map displacement: {np.max(np.abs(disp_x_blended)):.2f}px")
print(f"Mean displacement where mask > 0: {np.mean(np.abs(disp_x_blended)[mask > 0]):.2f}px")

# Check inside the mask where the nostrils are:
# What is the actual shift in pixels?
warped = cv2.remap(img, np.ascontiguousarray(map_x_blended), np.ascontiguousarray(yy), interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)

# Now check alpha blending:
alpha = support[..., None]
result = (warped.astype(np.float32) * alpha + img.astype(np.float32) * (1.0 - alpha))
result = np.clip(result, 0, 255).astype(np.uint8)

diff_warped = cv2.absdiff(img, warped)
diff_final = cv2.absdiff(img, result)

print(f"Max diff of warped: {np.max(diff_warped)}, mean inside mask: {np.mean(diff_warped[mask > 0]):.2f}")
print(f"Max diff of final: {np.max(diff_final)}, mean inside mask: {np.mean(diff_final[mask > 0]):.2f}")

