import hashlib
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, ".")
from deformation.engine import deformation_engine
from deformation.masking import load_nose_mask
from generation.viewpoints import CLINICAL_VIEWPOINTS

views = [v.key for v in CLINICAL_VIEWPOINTS]
orig_dir = Path("storage/generated/P01/original")
def_dir = Path("storage/deformations/7/P01/broad")
diag_dir = Path("storage/deformations/diagnostics_broad_audit")
diag_dir.mkdir(parents=True, exist_ok=True)

metrics_report = {}

print(f"{'Viewpoint':<12} | {'Orig Size':<10} | {'Def Size':<10} | {'Orig SHA':<10} | {'Def SHA':<10} | {'MaxDisp':<8} | {'MeanDisp':<9} | {'MaxDiff':<8} | {'MeanDiff':<9} | {'Changed%':<9}")
print("-" * 110)

for vp in views:
    orig_file = orig_dir / f"{vp}.png"
    def_file = def_dir / f"{vp}_broad.png"
    mask_file = def_dir / f"{vp}_mask.png"
    
    if not orig_file.exists():
        print(f"Missing original file: {orig_file}")
        continue
        
    orig_bytes = orig_file.read_bytes()
    orig_sha = hashlib.sha256(orig_bytes).hexdigest()[:8]
    orig_size = len(orig_bytes)
    
    orig_img = cv2.imread(str(orig_file))
    h, w = orig_img.shape[:2]
    
    # Load mask
    mask = load_nose_mask(mask_file, (h, w)) if mask_file.exists() else None
    if mask is None:
        ref_path = Path(f"deformation/reference_masks/{vp}/nose_binary.png")
        mask = load_nose_mask(ref_path, (h, w))
        
    mask_pixels = int(np.count_nonzero(mask))
    mask_pct = (mask_pixels / (h * w)) * 100.0
    
    # Run deformation engine to capture exact displacement maps and output
    res = deformation_engine.apply_deformation(
        image_bgr=orig_img,
        nose_mask=mask,
        viewpoint=vp,
        deformation_type="broad",
        strength=0.45,
    )
    
    deformed_img = res.output_image
    
    # If saved deformed file exists, check its SHA and size
    if def_file.exists():
        def_bytes = def_file.read_bytes()
        def_sha = hashlib.sha256(def_bytes).hexdigest()[:8]
        def_size = len(def_bytes)
    else:
        cv2.imwrite(str(def_file), deformed_img)
        def_bytes = def_file.read_bytes()
        def_sha = hashlib.sha256(def_bytes).hexdigest()[:8]
        def_size = len(def_bytes)
        
    # Difference calculations
    diff = cv2.absdiff(orig_img, deformed_img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    max_diff = int(np.max(diff))
    mean_diff = float(np.mean(diff))
    
    inside_mask = mask > 0
    mean_inside = float(np.mean(diff[inside_mask])) if np.any(inside_mask) else 0.0
    mean_outside = float(np.mean(diff[~inside_mask])) if np.any(~inside_mask) else 0.0
    
    changed_pixels = int(np.count_nonzero(diff_gray > 0))
    changed_pct = (changed_pixels / (h * w)) * 100.0
    
    # Reconstruct displacement map
    # We can measure displacement by optical flow or by checking algorithm internals
    # For _broad_nose_warp:
    ys, xs = np.where(inside_mask)
    if len(xs) > 0:
        y_top, y_bottom = int(ys.min()), int(ys.max())
        x_left, x_right = int(xs.min()), int(xs.max())
        nose_h = y_bottom - y_top + 1
        nose_w = x_right - x_left + 1
        
        centerline = np.full(h, np.nan, dtype=np.float32)
        for y in range(y_top, y_bottom + 1):
            rx = np.flatnonzero(inside_mask[y])
            if rx.size:
                centerline[y] = (rx.min() + rx.max()) / 2.0
        vrows = np.flatnonzero(np.isfinite(centerline))
        centerline = np.interp(np.arange(h), vrows, centerline[vrows]).astype(np.float32)
        centerline = cv2.GaussianBlur(centerline.reshape(-1, 1), (0, 0), sigmaX=max(1.0, nose_h * 0.012)).ravel()
        
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = np.clip((np.arange(h, dtype=np.float32) - y_top) / max(nose_h - 1, 1), 0.0, 1.0)
        
        view_factor = 1.0 if vp == "frontal" else (0.85 if vp == "basal" else (0.60 if vp == "helicopter" else 0.65))
        lower_c = 0.70 if vp == "frontal" else (0.60 if vp == "basal" else (0.55 if vp == "helicopter" else 0.66))
        lower_w = 0.25 if vp == "frontal" else (0.30 if vp in ("basal", "helicopter") else 0.23)
        
        v_prof = np.exp(-0.5 * ((t - lower_c) / lower_w) ** 2) * (np.clip(np.sin(np.pi * t), 0.0, 1.0) ** 0.7)
        v_prof *= ((np.arange(h) >= y_top) & (np.arange(h) <= y_bottom))
        
        ksize = max(5, int(round(nose_w * 0.12)) | 1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        expanded = cv2.dilate(inside_mask.astype(np.uint8), kern, iterations=1)
        support = cv2.GaussianBlur(expanded.astype(np.float32), (0, 0), sigmaX=max(1.0, nose_w * 0.025))
        support = np.clip(support, 0.0, 1.0)
        
        h_dist = xx - centerline[:, None]
        amount = 0.45 * view_factor * v_prof[:, None]
        
        map_x = (centerline[:, None] + h_dist / (1.0 + amount)).astype(np.float32)
        map_x = xx + (map_x - xx) * support
        disp_mag = np.abs(map_x - xx)
    else:
        disp_mag = np.zeros((h, w), dtype=np.float32)
        
    max_disp = float(np.max(disp_mag))
    mean_disp = float(np.mean(disp_mag[inside_mask])) if np.any(inside_mask) else 0.0
    nonzero_disp_pixels = int(np.count_nonzero(disp_mag > 0.1))
    
    print(f"{vp:<12} | {orig_size:<10} | {def_size:<10} | {orig_sha:<10} | {def_sha:<10} | {max_disp:<8.2f} | {mean_disp:<9.2f} | {max_diff:<8} | {mean_diff:<9.4f} | {changed_pct:<9.2f}%")
    
    # Generate 7 diagnostic visual components:
    # 1. Original FLUX
    p1 = orig_img.copy()
    # 2. Mask (3-channel)
    p2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # 3. Mask overlay
    p3 = orig_img.copy()
    p3[inside_mask] = (p3[inside_mask] * 0.45 + np.array([0, 220, 0]) * 0.55).astype(np.uint8)
    # 4. Displacement field visualization (Jet colormap)
    disp_norm = np.clip(disp_mag / max(max_disp, 1.0) * 255.0, 0, 255).astype(np.uint8)
    p4 = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    p4[disp_mag < 0.05] = orig_img[disp_mag < 0.05] // 3
    # 5. Deformed image
    p5 = deformed_img.copy()
    # 6. Absolute difference image (amplified 10x with HOT colormap)
    diff_amp = np.clip(diff_gray.astype(np.float32) * 8.0, 0, 255).astype(np.uint8)
    p6 = cv2.applyColorMap(diff_amp, cv2.COLORMAP_HOT)
    # 7. Zoomed comparison of nose region
    if len(xs) > 0:
        pad_y = int(nose_h * 0.3)
        pad_x = int(nose_w * 0.3)
        y0, y1 = max(0, y_top - pad_y), min(h, y_bottom + pad_y)
        x0, x1 = max(0, x_left - pad_x), min(w, x_right + pad_x)
        crop_orig = cv2.resize(orig_img[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LINEAR)
        crop_def = cv2.resize(deformed_img[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LINEAR)
        # Split vertical crop
        p7 = np.zeros_like(orig_img)
        split_mid = w // 2
        p7[:, :split_mid] = crop_orig[:, :split_mid]
        p7[:, split_mid:] = crop_def[:, split_mid:]
        cv2.line(p7, (split_mid, 0), (split_mid, h), (0, 255, 255), 2)
        cv2.putText(p7, "ZOOM: Orig", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(p7, "ZOOM: Deformed", (split_mid + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        p7 = deformed_img.copy()
        
    # Add title headers to each panel
    def add_title(img_in, text):
        res = img_in.copy()
        cv2.rectangle(res, (0, 0), (w, 30), (15, 15, 20), -1)
        cv2.putText(res, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        return res
        
    t1 = add_title(p1, f"1. Orig FLUX: {vp}")
    t2 = add_title(p2, f"2. Mask ({mask_pixels} px, {mask_pct:.1f}%)")
    t3 = add_title(p3, f"3. Mask Alignment Overlay")
    t4 = add_title(p4, f"4. Displacement Map (max:{max_disp:.1f}px)")
    t5 = add_title(p5, f"5. Deformed: BROAD NOSE")
    t6 = add_title(p6, f"6. Pixel Diff (amp 8x, max:{max_diff})")
    t7 = add_title(p7, f"7. Zoom Split Comparison")
    
    # Combine into 1 row of 7 panels
    combined = np.hstack([t1, t2, t3, t4, t5, t6, t7])
    panel_path = diag_dir / f"diagnostic_{vp}_broad.png"
    cv2.imwrite(str(panel_path), combined)
    
    metrics_report[vp] = {
        "viewpoint": vp,
        "dimensions": f"{w}x{h}",
        "orig_size_bytes": orig_size,
        "def_size_bytes": def_size,
        "orig_sha256": orig_sha,
        "def_sha256": def_sha,
        "is_sha_different": orig_sha != def_sha,
        "mask_pixels": mask_pixels,
        "mask_pct_image": round(mask_pct, 2),
        "max_displacement_px": round(max_disp, 2),
        "mean_displacement_px": round(mean_disp, 2),
        "nonzero_displacement_pixels": nonzero_disp_pixels,
        "max_pixel_diff": max_diff,
        "mean_pixel_diff_total": round(mean_diff, 4),
        "mean_pixel_diff_inside_mask": round(mean_inside, 4),
        "mean_pixel_diff_outside_mask": round(mean_outside, 4),
        "changed_pixels_count": changed_pixels,
        "changed_pct_total": round(changed_pct, 2),
        "diagnostic_panel_path": str(panel_path),
    }

report_json_path = diag_dir / "audit_metrics.json"
report_json_path.write_text(json.dumps(metrics_report, indent=2))
print(f"\nAll 9 diagnostic comparison panels saved to {diag_dir}/")

