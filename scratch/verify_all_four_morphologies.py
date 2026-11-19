import sys
sys.path.insert(0, ".")
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

from deformation.engine import deformation_engine
from deformation.masking import load_nose_mask, prepare_nose_mask
from generation.viewpoints import CLINICAL_VIEWPOINTS

orig_dir = Path("storage/generated/P01/original")
ref_masks_dir = Path("deformation/reference_masks")

morphologies = [
    ("saddle", 0.60, None),
    ("hump", 0.45, None),
    ("broad", 0.45, None),
    ("deviated", 0.40, "right"),
]

views = [v.key for v in CLINICAL_VIEWPOINTS]

results = {}

for morph, strength, direction in morphologies:
    results[morph] = {}
    print(f"\n=================== MORPHOLOGY: {morph.upper()} (strength={strength}, dir={direction}) ===================")
    print(f"{'Viewpoint':<12} | {'Dims':<9} | {'Orig SHA':<10} | {'Def SHA':<10} | {'MaxDiff':<7} | {'MeanDiff':<8} | {'ChgPx':<6} | {'ChgPct':<7} | {'Preserved?':<10} | {'Status':<15}")
    print("-" * 115)
    
    for v in views:
        img_path = orig_dir / f"{v}.png"
        assert img_path.exists(), f"Original image {img_path} missing!"
        
        orig_bgr = cv2.imread(str(img_path))
        orig_sha = hashlib.sha256(img_path.read_bytes()).hexdigest()[:8]
        h, w = orig_bgr.shape[:2]
        
        # Load canonical mask
        mask_path = ref_masks_dir / v / "nose_binary.png"
        if mask_path.exists():
            mask = load_nose_mask(mask_path, (h, w))
        else:
            dummy = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(dummy, (w // 2, int(h * 0.52)), (w // 10, h // 6), 0, 0, 360, 255, -1)
            mask = prepare_nose_mask(dummy, (h, w))
            
        res = deformation_engine.apply_deformation(
            image_bgr=orig_bgr,
            nose_mask=mask,
            viewpoint=v,
            deformation_type=morph,
            strength=strength,
            direction=direction,
        )
        
        assert res.success, f"Execution failed for {morph} on {v}: {res.error_message}"
        def_bgr = res.output_image
        
        # Calculate diff
        diff = cv2.absdiff(orig_bgr, def_bgr)
        max_d = int(np.max(diff))
        mean_d = float(np.mean(diff))
        chg_px = int(np.count_nonzero(diff > 0))
        chg_pct = (chg_px / (h * w)) * 100.0
        
        def_bytes = cv2.imencode(".png", def_bgr)[1].tobytes()
        def_sha = hashlib.sha256(def_bytes).hexdigest()[:8]
        
        # Clinically preserved check
        is_intentionally_preserved = (morph == "hump" and v in ("basal", "helicopter"))
        
        if max_d == 0 or chg_px < 10:
            status = "NO_VISIBLE_CHANGE" if is_intentionally_preserved else "NO_VISIBLE_CHANGE"
        else:
            status = "COMPLETED"
            
        print(f"{v:<12} | {f'{w}x{h}':<9} | {orig_sha:<10} | {def_sha:<10} | {max_d:<7} | {mean_d:<8.4f} | {chg_px:<6} | {chg_pct:<6.2f}% | {str(is_intentionally_preserved):<10} | {status:<15}")
        
        results[morph][v] = {
            "viewpoint": v,
            "dimensions": f"{w}x{h}",
            "orig_sha256": orig_sha,
            "def_sha256": def_sha,
            "max_pixel_diff": max_d,
            "mean_pixel_diff": round(mean_d, 4),
            "changed_pixels": chg_px,
            "changed_pct": round(chg_pct, 3),
            "intentionally_preserved": is_intentionally_preserved,
            "status": status,
        }

out_file = Path("storage/deformations/four_morphologies_audit.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved complete audit to {out_file}")
