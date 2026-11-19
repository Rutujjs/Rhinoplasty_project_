import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
audit_dir = BASE_DIR / "storage" / "diagnostics" / "mask_audit"
p01_dir = BASE_DIR / "storage" / "generated" / "P01" / "original"

views = ["frontal", "left_20", "left_40", "left_60", "right_20", "right_40", "right_60", "basal", "helicopter"]

print("=== CHECKING NOSTRIL AND EDGE ALIGNMENT WITH MASK ===")

for v in views:
    orig = cv2.imread(str(p01_dir / f"{v}.png"))
    mask = cv2.imread(str(audit_dir / f"mask_{v}.png"), cv2.IMREAD_GRAYSCALE)
    h, w = orig.shape[:2]
    
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    
    # In facial images, nostrils and eye corners have very dark pixels / strong edges
    # Find darkest 2% pixels in the midface region (likely nostrils or shadow)
    mid_y_start, mid_y_end = int(h * 0.25), int(h * 0.65)
    if v == "helicopter":
        mid_y_start, mid_y_end = int(h * 0.50), int(h * 0.85)
    elif v == "basal":
        mid_y_start, mid_y_end = int(h * 0.15), int(h * 0.45)
        
    midface_gray = gray[mid_y_start:mid_y_end, :]
    threshold_val = np.percentile(midface_gray, 5) # darkest 5%
    dark_features = (gray < threshold_val).astype(np.uint8) * 255
    dark_features[:mid_y_start, :] = 0
    dark_features[mid_y_end:, :] = 0
    
    # Overlap between mask and dark nasal features (e.g. nostrils / nasal shadow)
    mask_dark_overlap = np.count_nonzero((mask > 0) & (dark_features > 0))
    total_dark_in_midface = np.count_nonzero(dark_features)
    
    # Measure edges via Canny
    edges = cv2.Canny(gray, 50, 150)
    edges_in_mask = np.count_nonzero((mask > 0) & (edges > 0))
    
    # Mask centroid vs midface dark feature centroid
    ys_dark, xs_dark = np.where(dark_features > 0)
    if len(xs_dark) > 0:
        dark_cx = float(np.mean(xs_dark))
        dark_cy = float(np.mean(ys_dark))
    else:
        dark_cx, dark_cy = 0.0, 0.0
        
    ys_m, xs_m = np.where(mask > 0)
    mask_cx = float(np.mean(xs_m))
    mask_cy = float(np.mean(ys_m))
    
    offset_x = mask_cx - dark_cx
    offset_y = mask_cy - dark_cy
    
    print(f"{v:<12}: mask_center=({mask_cx:.1f}, {mask_cy:.1f}), dark_center=({dark_cx:.1f}, {dark_cy:.1f}), offset=(dx={offset_x:+.1f}, dy={offset_y:+.1f}), edges_in_mask={edges_in_mask}")

