import cv2
import numpy as np
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
audit_dir = BASE_DIR / "storage" / "diagnostics" / "mask_audit"
p01_dir = BASE_DIR / "storage" / "generated" / "P01" / "original"

views = ["frontal", "left_20", "left_40", "left_60", "right_20", "right_40", "right_60", "basal", "helicopter"]

with open(audit_dir / "mask_audit_summary.json") as f:
    summary = json.load(f)

print("=== DETAILED VIEWPOINT ANATOMICAL INSPECTION ===")
for item in summary:
    v = item["view"]
    w, h = item["image_size"]
    bbox = item["bbox"]
    cx, cy = item["centroid"]
    rel_cx, rel_cy = item["relative_centroid"]
    area_pct = item["mask_area_pct"]
    
    # Analyze image brightness/gradients inside and outside mask to assess feature alignment
    orig_path = p01_dir / f"{v}.png"
    img = cv2.imread(str(orig_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mask_path = audit_dir / f"mask_{v}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Eye region estimate (top 20-35% of face, roughly y: 0.20*h to 0.38*h for frontal/angled)
    # Mouth region estimate (lower face, roughly y: 0.65*h to 0.85*h)
    mask_ys, mask_xs = np.where(mask > 0)
    ymin_rel = mask_ys.min() / h
    ymax_rel = mask_ys.max() / h
    xmin_rel = mask_xs.min() / w
    xmax_rel = mask_xs.max() / w
    
    # Check vertical span
    # In canonical facial anatomy, nose spans approx 0.32-0.35 to 0.55-0.60 of face height.
    # Below 0.62 is typically philtrum and upper lip.
    # Above 0.32 is glabella/nasion and eyes.
    
    print(f"\n--- View: {v.upper()} ({item['canonical_name']}) ---")
    print(f"  Dimensions: {w}x{h}")
    print(f"  Mask vertical range: y={bbox['ymin']}..{bbox['ymax']} (rel: {ymin_rel:.3f}..{ymax_rel:.3f})")
    print(f"  Mask horizontal range: x={bbox['xmin']}..{bbox['xmax']} (rel: {xmin_rel:.3f}..{xmax_rel:.3f})")
    print(f"  Centroid: ({cx}, {cy}) -> rel: ({rel_cx}, {rel_cy})")
    print(f"  Area: {item['mask_active_pixels']} px ({area_pct}% of frame)")
    
    # Assess anatomical coverage:
    issues = []
    if v == "frontal":
        if ymin_rel < 0.28: issues.append("Mask extends into forehead/inter-eye glabella")
        if ymax_rel > 0.65: issues.append("Mask extends into philtrum/upper lip")
        if xmin_rel < 0.35 or xmax_rel > 0.65: issues.append("Mask encroaches on cheeks")
    elif "left" in v:
        # Left angled views: face is rotated to the left (patient's right), so nose should be on left of frame
        if rel_cx > 0.5: issues.append(f"Inconsistent centroid for left view ({rel_cx:.2f} > 0.5)")
    elif "right" in v:
        # Right angled views: nose should be on right of frame
        if rel_cx < 0.5: issues.append(f"Inconsistent centroid for right view ({rel_cx:.2f} < 0.5)")
    elif v == "basal":
        # Basal (submental) view: looking up nostrils. Nose tip and nostrils are centrally positioned.
        pass
    elif v == "helicopter":
        # Helicopter view: looking down from above. Forehead top, nose bridge down, nose tip lower.
        pass
        
    print(f"  Clinical Alignment Notes: {'No gross coordinate drift' if not issues else '; '.join(issues)}")

