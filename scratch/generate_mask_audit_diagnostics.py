import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from deformation.engine import deformation_engine, normalize_viewpoint_key
from deformation.masking import load_nose_mask, clean_nose_mask, prepare_nose_mask, make_overlay
from backend.app.services.deformation_service import deformation_service

out_dir = BASE_DIR / "storage" / "diagnostics" / "mask_audit"
out_dir.mkdir(parents=True, exist_ok=True)

p01_dir = BASE_DIR / "storage" / "generated" / "P01" / "original"

views = [
    ("frontal", "FRONT_0"),
    ("left_20", "LEFT_20"),
    ("left_40", "LEFT_40"),
    ("left_60", "LEFT_60"),
    ("right_20", "RIGHT_20"),
    ("right_40", "RIGHT_40"),
    ("right_60", "RIGHT_60"),
    ("basal", "BASAL"),
    ("helicopter", "HELICOPTER"),
]

print(f"Generating 9-view mask diagnostics into {out_dir}...")

audit_records = []

for view_key, view_display in views:
    orig_path = p01_dir / f"{view_key}.png"
    if not orig_path.exists():
        print(f"Original image not found: {orig_path}")
        continue
    
    orig_bgr = cv2.imread(str(orig_path))
    h, w = orig_bgr.shape[:2]
    
    # Create dummy image record to pass to _resolve_or_create_mask
    class DummyRecord:
        mask_image_path = None
    
    # Resolve mask using the exact service method
    resolved_mask = deformation_service._resolve_or_create_mask(
        image_record=DummyRecord(),
        image_shape=(h, w),
        view=view_key,
        patient_id="P01"
    )
    
    # Create red semi-transparent overlay
    overlay_bgr = make_overlay(orig_bgr, resolved_mask)
    
    # Execute Broad and Saddle deformation to test
    res_broad = deformation_engine.apply_deformation(
        image_bgr=orig_bgr,
        nose_mask=resolved_mask,
        viewpoint=view_key,
        deformation_type="broad",
        strength=0.45
    )
    
    res_saddle = deformation_engine.apply_deformation(
        image_bgr=orig_bgr,
        nose_mask=resolved_mask,
        viewpoint=view_key,
        deformation_type="saddle",
        strength=0.65
    )
    
    # 4-panel image: [Original, Binary Mask (3-ch), Overlay, Deformed (Broad)]
    mask_3ch = cv2.cvtColor(resolved_mask, cv2.COLOR_GRAY2BGR)
    deformed_img = res_broad.output_image if res_broad.success else orig_bgr
    
    # Annotate panels with headers
    panel_h = h
    header_h = 40
    total_h = panel_h + header_h
    
    def create_annotated_panel(img, title):
        p = np.zeros((total_h, w, 3), dtype=np.uint8)
        p[:header_h, :] = (24, 24, 27) # Dark header
        cv2.putText(p, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        p[header_h:, :] = img
        return p
    
    p1 = create_annotated_panel(orig_bgr, f"1. Original ({w}x{h})")
    p2 = create_annotated_panel(mask_3ch, "2. Binary Nasal Mask")
    p3 = create_annotated_panel(overlay_bgr, "3. Mask Overlay (Red)")
    p4 = create_annotated_panel(deformed_img, "4. Deformed (Broad 0.45)")
    
    strip = np.hstack([p1, p2, p3, p4])
    
    strip_out_path = out_dir / f"diagnostic_{view_key}.png"
    cv2.imwrite(str(strip_out_path), strip)
    
    # Also save individual binary mask and overlay for detailed inspection
    cv2.imwrite(str(out_dir / f"mask_{view_key}.png"), resolved_mask)
    cv2.imwrite(str(out_dir / f"overlay_{view_key}.png"), overlay_bgr)
    
    # Calculate mask stats
    active_px = int(np.count_nonzero(resolved_mask))
    ys, xs = np.where(resolved_mask > 0)
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    box_w, box_h = xmax - xmin, ymax - ymin
    cx, cy = float((xmin + xmax) / 2), float((ymin + ymax) / 2)
    
    diff = cv2.absdiff(orig_bgr, deformed_img)
    modified_px = int(np.count_nonzero(np.any(diff > 2, axis=2)))
    
    # Check if pixels outside mask were modified (leakage test)
    outside_mask = (resolved_mask == 0)
    leakage_px = int(np.count_nonzero(np.any(diff[outside_mask] > 2, axis=1)))
    
    record = {
        "view": view_key,
        "canonical_name": view_display,
        "image_size": [w, h],
        "mask_active_pixels": active_px,
        "mask_area_pct": round(active_px / (w * h) * 100, 2),
        "bbox": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "w": box_w, "h": box_h},
        "centroid": [round(cx, 1), round(cy, 1)],
        "relative_centroid": [round(cx / w, 3), round(cy / h, 3)],
        "broad_modified_pixels": modified_px,
        "leakage_outside_mask": leakage_px,
        "strip_image": str(strip_out_path)
    }
    audit_records.append(record)
    print(f"[{view_display:<10}] Mask area={active_px}px ({record['mask_area_pct']}%), Centroid=({cx:.1f}, {cy:.1f}), ModPx={modified_px}, Leakage={leakage_px}")

import json
with open(out_dir / "mask_audit_summary.json", "w") as f:
    json.dump(audit_records, f, indent=2)

print("\nAudit diagnostics complete. Summary saved to storage/diagnostics/mask_audit/mask_audit_summary.json")

