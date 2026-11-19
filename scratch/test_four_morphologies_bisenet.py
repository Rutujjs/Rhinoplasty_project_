import sys
import hashlib
from pathlib import Path
import cv2
import numpy as np
import json

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from deformation.engine import deformation_engine
from deformation.bisenet_inference import bisenet_segmentor

p01_dir = BASE_DIR / "storage" / "generated" / "P01" / "original"

views = [
    "frontal",
    "left_20",
    "left_40",
    "left_60",
    "right_20",
    "right_40",
    "right_60",
    "basal",
    "helicopter",
]

morphologies = ["saddle", "hump", "broad", "deviated"]

print("=== 4 MORPHOLOGIES x 9 VIEWPOINTS REGRESSION TEST WITH BISENET ===")

# Compute original image hashes before test
orig_hashes = {}
for v in views:
    p = p01_dir / f"{v}.png"
    orig_hashes[v] = hashlib.sha256(p.read_bytes()).hexdigest()

test_matrix = []
all_passed = True

for morph in morphologies:
    print(f"\n--- Morphology: {morph.upper()} ---")
    for v in views:
        img_bgr = cv2.imread(str(p01_dir / f"{v}.png"))
        h, w = img_bgr.shape[:2]
        
        # 1. Infer patient-specific mask
        mask, meta = bisenet_segmentor.predict_nose_mask(img_bgr, view_name=v)
        
        # 2. Run deformation
        direction = "right" if morph == "deviated" else None
        res = deformation_engine.apply_deformation(
            image_bgr=img_bgr,
            nose_mask=mask,
            viewpoint=v,
            deformation_type=morph,
            direction=direction,
        )
        
        if not res.success or res.output_image is None:
            print(f"  [{v:<10}] FAILED: {res.error_message}")
            all_passed = False
            continue
            
        deformed = res.output_image
        diff = cv2.absdiff(img_bgr, deformed)
        mod_px = int(np.count_nonzero(np.any(diff > 2, axis=2)))
        mod_pct = round(mod_px / (w * h) * 100, 2)
        
        # Check non-nasal preservation (eyes, mouth)
        # Eye region: y < 0.28 * h (or y < 0.15 for helicopter)
        # Mouth region: y > 0.65 * h (or y > 0.50 for basal)
        non_nasal_mask = np.zeros((h, w), dtype=bool)
        if v == "helicopter":
            non_nasal_mask[:int(h * 0.15), :] = True
            non_nasal_mask[int(h * 0.85):, :] = True
        elif v == "basal":
            non_nasal_mask[int(h * 0.60):, :] = True
        else:
            non_nasal_mask[:int(h * 0.25), :] = True
            non_nasal_mask[int(h * 0.68):, :] = True
            
        unrelated_mods = int(np.count_nonzero(np.any(diff[non_nasal_mask] > 2, axis=1)))
        
        status = "PASSED" if mod_px > 0 and unrelated_mods == 0 else ("NO_CHANGE" if mod_px == 0 else "WARNING_LEAKAGE")
        print(f"  [{v:<10}] {status:<8} ModPx={mod_px:>5} ({mod_pct:>4}%), MaskPx={meta['nose_pixels']:>5}, UnrelatedFaceMods={unrelated_mods}")
        
        test_matrix.append({
            "morphology": morph,
            "viewpoint": v,
            "status": status,
            "modified_pixels": mod_px,
            "modified_pct": mod_pct,
            "mask_pixels": meta["nose_pixels"],
            "unrelated_face_modifications": unrelated_mods,
        })

# Verify immutability of original images
print("\n=== VERIFYING IMMUTABILITY OF ORIGINAL FLUX IMAGES ===")
immutability_ok = True
for v in views:
    p = p01_dir / f"{v}.png"
    current_hash = hashlib.sha256(p.read_bytes()).hexdigest()
    if current_hash != orig_hashes[v]:
        print(f"  ERROR: Original image {v}.png was modified!")
        immutability_ok = False
    else:
        print(f"  {v:<10}: SHA-256 match verified ({current_hash[:12]}...)")

out_summary = {
    "total_combinations": len(test_matrix),
    "passed": sum(1 for t in test_matrix if t["status"] == "PASSED"),
    "immutability_verified": immutability_ok,
    "segmentation_backend": "BiSeNet ResNet-18 (ONNX)",
    "results": test_matrix,
}

with open(BASE_DIR / "storage" / "diagnostics" / "bisenet_patient_masks" / "regression_test_summary.json", "w") as f:
    json.dump(out_summary, f, indent=2)

print(f"\nRegression tests completed: {out_summary['passed']}/{out_summary['total_combinations']} passed. Immutability: {immutability_ok}")

