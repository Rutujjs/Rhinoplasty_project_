import sys
sys.path.insert(0, ".")
from pathlib import Path
import cv2
import numpy as np
from deformation.engine import deformation_engine
from deformation.masking import load_nose_mask

orig_path = Path("storage/generated/P01/original/frontal.png")
mask_path = Path("storage/deformations/7/P01/broad/frontal_mask.png")

orig = cv2.imread(str(orig_path))
mask = load_nose_mask(mask_path, orig.shape[:2])

print(f"{'Morphology':<12} | {'Strength':<10} | {'Changed Pct':<12} | {'Max Diff':<10} | {'Mean Diff':<12} | {'Mean Diff (inside)':<20}")
print("-" * 85)

for morph in ["saddle", "hump", "broad", "deviated"]:
    for s in [0.25, 0.45, 0.60, 0.80]:
        res = deformation_engine.apply_deformation(
            image_bgr=orig,
            nose_mask=mask,
            viewpoint="frontal",
            deformation_type=morph,
            strength=s,
            direction="right",
        )
        def_img = res.output_image
        diff = cv2.absdiff(orig, def_img)
        diff_inside = diff[mask > 0]
        
        changed = np.count_nonzero(np.any(diff > 0, axis=-1))
        pct = changed / orig.shape[0] / orig.shape[1] * 100
        max_d = np.max(diff)
        mean_d = np.mean(diff)
        mean_inside = np.mean(diff_inside)
        
        print(f"{morph:<12} | {s:<10.2f} | {pct:<12.2f}% | {max_d:<10} | {mean_d:<12.4f} | {mean_inside:<20.4f}")
