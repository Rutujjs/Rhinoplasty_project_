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

# Let's inspect step-by-step what happens in deviated_nose_deformation
centerline = np.full(h, np.nan, dtype=np.float32)
for y in range(y_top, y_bottom + 1):
    row_x = np.flatnonzero(mask[y])
    if row_x.size:
        centerline[y] = (float(row_x.min()) + float(row_x.max())) / 2.0
valid_rows = np.flatnonzero(np.isfinite(centerline))
centerline = np.interp(np.arange(h), valid_rows, centerline[valid_rows]).astype(np.float32)
centerline = cv2.GaussianBlur(centerline.reshape(-1, 1), (0, 0), sigmaX=max(1.0, nose_height * 0.015 if 'nose_height' in locals() else nose_h * 0.015)).ravel()

t = np.clip((np.arange(h, dtype=np.float32) - y_top) / max(nose_h - 1, 1), 0.0, 1.0)
sine_profile = np.clip(np.sin(np.pi * t), 0.0, 1.0)
vertical_profile = np.power(sine_profile, 1.2).astype(np.float32)
vertical_profile[:y_top] = 0.0
vertical_profile[y_bottom + 1:] = 0.0

yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
horizontal_distance = xx - centerline[:, None]
sigma_x = max(2.0, nose_w * 0.45)
horizontal_profile = np.exp(-0.5 * (horizontal_distance / sigma_x) ** 2).astype(np.float32)

strength = 0.35
direction = "right"
sign = 1.0 if direction == "right" else -1.0
max_displacement = strength * nose_w * 0.30
displacement = (sign * max_displacement * vertical_profile[:, None] * horizontal_profile).astype(np.float32)

print(f"1. Theoretical max_displacement: {max_displacement:.2f} px")
print(f"2. Raw displacement in mask: min={displacement[mask>0].min():.2f}, max={displacement[mask>0].max():.2f}, mean={displacement[mask>0].mean():.2f} px")

# Feathering alpha
distance_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
feather_width = max(2.0, nose_w * 0.12)
alpha = np.clip(distance_inside / feather_width, 0.0, 1.0).astype(np.float32)
alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(0.8, nose_w * 0.015))
alpha *= mask.astype(np.float32)

print(f"3. Alpha in mask: min={alpha[mask>0].min():.2f}, max={alpha[mask>0].max():.2f}, mean={alpha[mask>0].mean():.2f}")

map_x = np.clip(xx - displacement, 0.0, float(w - 1)).astype(np.float32)
map_y = yy.copy()

x0 = np.floor(map_x).astype(np.int32)
y0 = np.floor(map_y).astype(np.int32)
x1 = np.minimum(x0 + 1, w - 1)
y1 = np.minimum(y0 + 1, h - 1)
source_valid = (
    (mask[y0, x0] > 0)
    & (mask[y0, x1] > 0)
    & (mask[y1, x0] > 0)
    & (mask[y1, x1] > 0)
)
invalid_mapping = (mask > 0) & (~source_valid)
cancelled_pixels = np.count_nonzero(invalid_mapping)
print(f"4. Invalid mapping (cancelled pixels inside mask): {cancelled_pixels} / {np.count_nonzero(mask)} ({cancelled_pixels / np.count_nonzero(mask)*100:.1f}%)")

map_x[invalid_mapping] = xx[invalid_mapping]
effective_disp = np.abs(xx - map_x)
print(f"5. Effective displacement after invalid_mapping cancellation: max={effective_disp[mask>0].max():.2f}, mean={effective_disp[mask>0].mean():.2f} px")

# Effective shift considering alpha blend:
effective_shift_with_alpha = effective_disp * alpha
print(f"6. Effective shift AFTER alpha blending: max={effective_shift_with_alpha[mask>0].max():.2f}, mean={effective_shift_with_alpha[mask>0].mean():.2f} px")

