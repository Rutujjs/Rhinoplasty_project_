import os
import sys
from pathlib import Path
import cv2
import numpy as np
import json

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from deformation.engine import deformation_engine
from deformation.masking import clean_nose_mask, make_overlay
from deformation.bisenet_inference import bisenet_segmentor, CELEBAMASK_HQ_CLASSES

out_dir = BASE_DIR / "storage" / "diagnostics" / "bisenet_patient_masks"
out_dir.mkdir(parents=True, exist_ok=True)

p01_dir = BASE_DIR / "storage" / "generated" / "P01" / "original"
ref_dir = BASE_DIR / "deformation" / "reference_masks"

views = [
    ("frontal", "FRONT_0", "frontal_0"),
    ("left_20", "LEFT_20", "left20"),
    ("left_40", "LEFT_40", "left40"),
    ("left_60", "LEFT_60", "left60"),
    ("right_20", "RIGHT_20", "right20"),
    ("right_40", "RIGHT_40", "right40"),
    ("right_60", "RIGHT_60", "right60"),
    ("basal", "BASAL", "basal_view"),
    ("helicopter", "HELICOPTER", "helicopter_view"),
]

# Color palette for 19 CelebAMask classes (BGR)
PALETTE = [
    (0, 0, 0),        # 0: background
    (204, 204, 255),  # 1: skin (light peach)
    (0, 102, 204),    # 2: l_brow
    (0, 102, 204),    # 3: r_brow
    (255, 0, 0),      # 4: l_eye (blue)
    (255, 0, 0),      # 5: r_eye (blue)
    (0, 255, 255),    # 6: eye_g (yellow)
    (204, 153, 255),  # 7: l_ear
    (204, 153, 255),  # 8: r_ear
    (153, 0, 153),    # 9: ear_r
    (0, 0, 255),      # 10: nose (bright red)
    (255, 255, 0),    # 11: mouth (cyan)
    (0, 153, 76),     # 12: u_lip (green)
    (0, 204, 102),    # 13: l_lip (light green)
    (102, 255, 255),  # 14: neck
    (204, 204, 0),    # 15: neck_l
    (128, 128, 128),  # 16: cloth (gray)
    (51, 102, 0),     # 17: hair (dark olive)
    (255, 128, 0),    # 18: hat (orange)
]

def colorize_parsing(labels):
    h, w = labels.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(min(len(PALETTE), 19)):
        colored[labels == class_id] = PALETTE[class_id]
    return colored

print("=== GENERATING 5-PANEL DIAGNOSTICS & COMPARISONS ===")
summary_results = []

for view_key, view_display, ref_alias in views:
    orig_path = p01_dir / f"{view_key}.png"
    if not orig_path.exists():
        continue
        
    orig_bgr = cv2.imread(str(orig_path))
    h, w = orig_bgr.shape[:2]
    
    # 1. BiSeNet inference: full 19-class parsing
    labels = bisenet_segmentor.predict_full_parsing(orig_bgr)
    colorized_parse = colorize_parsing(labels)
    
    # 2. Extract and clean patient-specific nasal mask
    bisenet_mask, meta = bisenet_segmentor.predict_nose_mask(orig_bgr, view_name=view_key)
    
    # 3. Semi-transparent mask overlay
    bisenet_overlay = make_overlay(orig_bgr, bisenet_mask)
    
    # 4. Deformed result (Broad nose)
    res_broad = deformation_engine.apply_deformation(
        image_bgr=orig_bgr,
        nose_mask=bisenet_mask,
        viewpoint=view_key,
        deformation_type="broad",
        strength=0.45
    )
    deformed_img = res_broad.output_image if res_broad.success else orig_bgr
    
    # 5. Load old static reference mask for side-by-side comparison
    ref_mask_file = ref_dir / ref_alias / "nose_binary.png"
    if ref_mask_file.exists():
        ref_mask_raw = cv2.imread(str(ref_mask_file), cv2.IMREAD_GRAYSCALE)
        ref_mask_resized = cv2.resize(ref_mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
        ref_overlay = make_overlay(orig_bgr, ref_mask_resized)
        ref_px = int(np.count_nonzero(ref_mask_resized))
    else:
        ref_mask_resized = np.zeros((h, w), dtype=np.uint8)
        ref_overlay = orig_bgr.copy()
        ref_px = 0
        
    # Build 5-panel strip
    header_h = 36
    total_h = h + header_h
    
    def annotate(img, title):
        p = np.zeros((total_h, w, 3), dtype=np.uint8)
        p[:header_h, :] = (24, 24, 27)
        cv2.putText(p, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        p[header_h:, :] = img
        return p
        
    mask_3ch = cv2.cvtColor(bisenet_mask, cv2.COLOR_GRAY2BGR)
    
    p1 = annotate(orig_bgr, f"1. Original ({w}x{h})")
    p2 = annotate(colorized_parse, "2. BiSeNet 19-Class Parse")
    p3 = annotate(mask_3ch, f"3. BiSeNet Mask ({meta['nose_pixels']}px)")
    p4 = annotate(bisenet_overlay, "4. BiSeNet Overlay (Red)")
    p5 = annotate(deformed_img, "5. Deformed Result (Broad)")
    
    strip_5panel = np.hstack([p1, p2, p3, p4, p5])
    strip_path = out_dir / f"diagnostic_{view_key}_5panel.png"
    cv2.imwrite(str(strip_path), strip_5panel)
    
    # Build comparison strip: [Original] | [Old Reference Overlay] | [New BiSeNet Overlay] | [Overlay Difference]
    diff_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    # Green = BiSeNet only, Blue = Old Reference only, Yellow = Overlap
    overlap = (bisenet_mask > 0) & (ref_mask_resized > 0)
    bisenet_only = (bisenet_mask > 0) & (ref_mask_resized == 0)
    ref_only = (bisenet_mask == 0) & (ref_mask_resized > 0)
    
    diff_overlay[overlap] = (0, 255, 255)       # Yellow: mutual agreement
    diff_overlay[bisenet_only] = (0, 255, 0)    # Green: BiSeNet detected (new anatomical coverage)
    diff_overlay[ref_only] = (255, 0, 0)        # Blue: Old static template excess / misalignment
    
    diff_blend = orig_bgr.copy()
    any_m = (bisenet_mask > 0) | (ref_mask_resized > 0)
    diff_blend[any_m] = (0.45 * orig_bgr[any_m].astype(np.float32) + 0.55 * diff_overlay[any_m].astype(np.float32)).astype(np.uint8)
    
    c1 = annotate(orig_bgr, f"Original {view_display}")
    c2 = annotate(ref_overlay, f"Old Reference Mask ({ref_px}px)")
    c3 = annotate(bisenet_overlay, f"New BiSeNet Mask ({meta['nose_pixels']}px)")
    c4 = annotate(diff_blend, "Diff: Green=BiSeNet, Blue=Old, Yellow=Both")
    
    comp_strip = np.hstack([c1, c2, c3, c4])
    comp_path = out_dir / f"comparison_{view_key}_reference_vs_bisenet.png"
    cv2.imwrite(str(comp_path), comp_strip)
    
    # Save individual artifact files
    cv2.imwrite(str(out_dir / f"bisenet_mask_{view_key}.png"), bisenet_mask)
    cv2.imwrite(str(out_dir / f"bisenet_parse_{view_key}.png"), colorized_parse)
    cv2.imwrite(str(out_dir / f"bisenet_overlay_{view_key}.png"), bisenet_overlay)
    cv2.imwrite(str(out_dir / f"deformed_broad_{view_key}.png"), deformed_img)
    
    # Calculate agreement metrics (IoU, Dice)
    intersection = int(np.count_nonzero(overlap))
    union = int(np.count_nonzero(any_m))
    iou = round(float(intersection / union), 3) if union > 0 else 0.0
    dice = round(float(2 * intersection / (meta["nose_pixels"] + ref_px)), 3) if (meta["nose_pixels"] + ref_px) > 0 else 0.0
    
    rec = {
        "view": view_key,
        "canonical_name": view_display,
        "dimensions": [w, h],
        "bisenet_pixels": meta["nose_pixels"],
        "bisenet_area_pct": meta["nose_area_percentage"],
        "bisenet_centroid": meta["centroid"],
        "bisenet_bbox": meta["bounding_box"],
        "reference_pixels": ref_px,
        "intersection_pixels": intersection,
        "bisenet_unique_pixels": int(np.count_nonzero(bisenet_only)),
        "reference_unique_pixels": int(np.count_nonzero(ref_only)),
        "iou_agreement": iou,
        "dice_similarity": dice,
        "strip_5panel_path": str(strip_path),
        "comparison_path": str(comp_path),
    }
    summary_results.append(rec)
    print(f"[{view_display:<10}] BiSeNet={meta['nose_pixels']}px, Ref={ref_px}px, IoU={iou}, Dice={dice}, BiSeNetUnique={rec['bisenet_unique_pixels']}")

with open(out_dir / "bisenet_vs_reference_comparison.json", "w") as f:
    json.dump(summary_results, f, indent=2)

print("\nAll 9 viewpoints diagnostic validation complete! Saved to storage/diagnostics/bisenet_patient_masks/")

