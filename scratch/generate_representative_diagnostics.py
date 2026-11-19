import sys
sys.path.insert(0, ".")
from pathlib import Path
import cv2
import numpy as np

from deformation.engine import deformation_engine
from deformation.masking import load_nose_mask, prepare_nose_mask

orig_dir = Path("storage/generated/P01/original")
ref_masks_dir = Path("deformation/reference_masks")
out_dir = Path("storage/deformations/representative_diagnostics")
out_dir.mkdir(parents=True, exist_ok=True)

test_cases = [
    ("saddle", "left_20", 0.60, None, "Saddle Nose (Left 20 Profile)"),
    ("hump", "left_40", 0.45, None, "Hump Nose (Left 40 Dorsum)"),
    ("broad", "frontal", 0.45, None, "Broad Nose (Frontal Alar Base)"),
    ("deviated", "frontal", 0.40, "right", "Deviated Nose (Frontal Asymmetry)"),
]

for morph, vp, strength, direction, title in test_cases:
    img_path = orig_dir / f"{vp}.png"
    orig_bgr = cv2.imread(str(img_path))
    h, w = orig_bgr.shape[:2]
    
    norm_v = vp.replace("_", "")
    candidates = [
        ref_masks_dir / vp / "nose_binary.png",
        ref_masks_dir / norm_v / "nose_binary.png",
        ref_masks_dir / f"{norm_v}_view" / "nose_binary.png",
        ref_masks_dir / f"{norm_v}_0" / "nose_binary.png",
    ]
    mask = None
    for cand in candidates:
        if cand.exists():
            mask = load_nose_mask(cand, (h, w))
            break
    if mask is None:
        dummy = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(dummy, (w // 2, int(h * 0.52)), (w // 10, h // 6), 0, 0, 360, 255, -1)
        mask = prepare_nose_mask(dummy, (h, w))
    
    res = deformation_engine.apply_deformation(
        image_bgr=orig_bgr,
        nose_mask=mask,
        viewpoint=vp,
        deformation_type=morph,
        strength=strength,
        direction=direction,
    )
    def_bgr = res.output_image
    
    # 1. Mask overlay
    overlay = orig_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    # 2. Amplified diff (8x Hot)
    diff = cv2.absdiff(orig_bgr, def_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_amp = np.clip(diff_gray.astype(np.float32) * 8.0, 0, 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_amp, cv2.COLORMAP_HOT)
    
    # 3. Zoomed side-by-side nose crop
    ys, xs = np.where(mask > 0)
    if len(ys) > 0:
        ymin, ymax = max(0, ys.min() - 25), min(h, ys.max() + 25)
        xmin, xmax = max(0, xs.min() - 25), min(w, xs.max() + 25)
        crop_orig = cv2.resize(orig_bgr[ymin:ymax, xmin:xmax], (w // 2, h))
        crop_def = cv2.resize(def_bgr[ymin:ymax, xmin:xmax], (w // 2, h))
        crop_comp = np.hstack([crop_orig, crop_def])
        cv2.putText(crop_comp, "BEFORE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(crop_comp, "AFTER", (w // 2 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        crop_comp = np.zeros((h, w, 3), dtype=np.uint8)
        
    def label_panel(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        return out
        
    p1 = label_panel(orig_bgr, "1. FLUX Reference")
    p2 = label_panel(overlay, "2. Mask Alignment")
    p3 = label_panel(def_bgr, f"3. Deformed ({morph.upper()})")
    p4 = label_panel(diff_color, "4. Amplified Diff (8x)")
    p5 = crop_comp
    
    top_row = np.hstack([p1, p2, p3])
    # Bottom row resize p4 & p5 to match top row width
    tw = top_row.shape[1]
    p4_w = tw // 2
    p5_w = tw - p4_w
    p4_r = cv2.resize(p4, (p4_w, h))
    p5_r = cv2.resize(p5, (p5_w, h))
    bottom_row = np.hstack([p4_r, p5_r])
    
    full_panel = np.vstack([top_row, bottom_row])
    
    # Add title bar
    bar = np.zeros((50, full_panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"MORPHOLOGY AUDIT: {title.upper()} | View: {vp} | Strength: {strength}", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    final_img = np.vstack([bar, full_panel])
    
    save_path = out_dir / f"diagnostic_{morph}_{vp}.png"
    cv2.imwrite(str(save_path), final_img)
    print(f"Generated representative panel: {save_path.name}")

print("All 4 representative diagnostics generated successfully!")
