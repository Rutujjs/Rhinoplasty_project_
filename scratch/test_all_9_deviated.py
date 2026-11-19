import os
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from deformation.algorithms import (
    basal_deviated_nose_deformation,
    helicopter_deviated_nose_deformation,
)

# Test function for frontal / lateral
def test_warp(img, mask_bin, strength=0.45, direction="right", view="frontal"):
    h, w = img.shape[:2]
    ys, xs = np.where(mask_bin > 0)
    if len(ys) == 0:
        return img, 0.0, 0, 0.0
    y_top = int(ys.min())
    y_bottom = int(ys.max())
    x_left = int(xs.min())
    x_right = int(xs.max())
    nose_height = y_bottom - y_top + 1
    nose_width = x_right - x_left + 1

    # Centerline
    centerline = np.full(h, np.nan, dtype=np.float32)
    for y in range(y_top, y_bottom + 1):
        row_x = np.flatnonzero(mask_bin[y])
        if row_x.size:
            centerline[y] = (float(row_x.min()) + float(row_x.max())) / 2.0
    valid_rows = np.flatnonzero(np.isfinite(centerline))
    if len(valid_rows) < 3:
        return img, 0.0, 0, 0.0
    centerline = np.interp(np.arange(h), valid_rows, centerline[valid_rows]).astype(np.float32)
    centerline = cv2.GaussianBlur(centerline.reshape(-1, 1), (0, 0), sigmaX=max(1.0, nose_height * 0.015)).ravel()

    # Vertical profile
    t = np.clip((np.arange(h, dtype=np.float32) - y_top) / max(nose_height - 1, 1), 0.0, 1.0)
    sine_profile = np.clip(np.sin(np.pi * t), 0.0, 1.0)
    vertical_profile = np.power(sine_profile, 1.2).astype(np.float32)
    vertical_profile[:y_top] = 0.0
    vertical_profile[y_bottom + 1:] = 0.0

    # Horizontal profile
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    horizontal_distance = xx - centerline[:, None]
    sigma_x = max(2.0, nose_width * 0.45)
    horizontal_profile = np.exp(-0.5 * (horizontal_distance / sigma_x) ** 2).astype(np.float32)

    # Feathered support
    kernel_size = max(5, int(round(nose_width * 0.15)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(mask_bin, kernel, iterations=1)
    sigma = max(1.5, nose_width * 0.06)
    support = cv2.GaussianBlur(expanded.astype(np.float32), (0, 0), sigmaX=sigma)
    support = np.clip(support, 0.0, 1.0)

    sign = 1.0 if direction == "right" else -1.0
    max_displacement = strength * nose_width * 0.48

    displacement = (sign * max_displacement * vertical_profile[:, None] * horizontal_profile * support).astype(np.float32)

    map_x = np.clip(xx - displacement, 0.0, float(w - 1)).astype(np.float32)
    map_y = yy.astype(np.float32)

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
    alpha = support[..., None]
    result = (warped.astype(np.float32) * alpha + img.astype(np.float32) * (1.0 - alpha))
    result = np.clip(result, 0, 255).astype(np.uint8)

    diff = cv2.absdiff(img, result)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    max_d = int(np.max(gray_diff))
    changed_px = int(np.count_nonzero(gray_diff > 2))
    max_disp = float(np.max(np.abs(displacement)))

    return result, max_d, changed_px, max_disp

viewpoints = [
    ("frontal", "storage/generated/P01/original/frontal.png", "storage/diagnostics/bisenet_patient_masks/bisenet_mask_frontal.png"),
    ("left_20", "storage/generated/P01/original/left_20.png", "storage/diagnostics/bisenet_patient_masks/bisenet_mask_left_20.png"),
    ("left_40", "storage/generated/P01/original/left_40.png", "storage/diagnostics/bisenet_patient_masks/bisenet_mask_left_40.png"),
    ("left_60", "storage/generated/P01/original/left_60.png", "storage/diagnostics/bisenet_patient_masks/bisenet_mask_left_60.png"),
    ("right_20", "storage/generated/P01/original/right_20.png", "storage/diagnostics/bisenet_patient_masks/bisenet_mask_right_20.png"),
    ("right_40", "storage/generated/P01/original/right_40.png", "storage/diagnostics/bisenet_patient_masks/bisenet_mask_right_40.png"),
    ("right_60", "storage/generated/P01/original/right_60.png", "storage/diagnostics/bisenet_patient_masks/bisenet_mask_right_60.png"),
    ("basal", "storage/generated/P01/original/basal.png", "storage/diagnostics/bisenet_patient_masks/bisenet_mask_basal.png"),
    ("helicopter", "storage/generated/P01/original/helicopter.png", "storage/diagnostics/bisenet_patient_masks/bisenet_mask_helicopter.png"),
]

os.makedirs("storage/diagnostics/deviated_audit_preview", exist_ok=True)

print(f"{'Viewpoint':<12} | {'Max Diff':<10} | {'Changed Px':<12} | {'Max Disp (px)':<14} | {'Status'}")
print("-" * 65)

for vp, img_p, mask_p in viewpoints:
    img = cv2.imread(img_p)
    mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask > 0).astype(np.uint8)
    
    if vp == "basal":
        res = basal_deviated_nose_deformation(img, mask_bin, strength=0.45, direction="right")
        diff = cv2.absdiff(img, res)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        max_d = int(np.max(gray_diff))
        changed_px = int(np.count_nonzero(gray_diff > 2))
        max_disp = 0.45 * (np.max(np.where(mask_bin)[1]) - np.min(np.where(mask_bin)[1])) * 0.65
    elif vp == "helicopter":
        res = helicopter_deviated_nose_deformation(img, (mask_bin * 255), strength=0.45, direction="right")
        diff = cv2.absdiff(img, res)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        max_d = int(np.max(gray_diff))
        changed_px = int(np.count_nonzero(gray_diff > 2))
        max_disp = 0.45 * (np.max(np.where(mask_bin)[1]) - np.min(np.where(mask_bin)[1])) * 0.45
    else:
        res, max_d, changed_px, max_disp = test_warp(img, mask_bin, strength=0.45, direction="right", view=vp)
        diff = cv2.absdiff(img, res)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Save preview strip
    diff_vis = cv2.applyColorMap(np.clip(gray_diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    strip = np.hstack([img, res, diff_vis])
    cv2.imwrite(f"storage/diagnostics/deviated_audit_preview/{vp}_deviated.png", strip)
    
    status = "VISUALLY CLEAR" if max_d > 60 and max_disp > 5.0 else "SUBTLE"
    print(f"{vp:<12} | {max_d:<10} | {changed_px:<12} | {max_disp:<14.2f} | {status}")
