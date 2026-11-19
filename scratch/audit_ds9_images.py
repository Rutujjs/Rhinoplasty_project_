import cv2
import numpy as np
import hashlib
from pathlib import Path

p01_dir = Path("storage/generated/P01/original")
ds9_dir = Path("storage/deformations/9/P01/deviated")

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

print("=== DATASET 9 (DEVIATED) ORIGINAL vs DEFORMED FILE AUDIT ===")

for v in views:
    orig_path = p01_dir / f"{v}.png"
    def_path = ds9_dir / f"{v}_deviated.png"
    mask_path = ds9_dir / f"{v}_mask.png"
    
    if not orig_path.exists() or not def_path.exists():
        print(f"{v}: MISSING FILE! orig={orig_path.exists()}, def={def_path.exists()}")
        continue
        
    orig = cv2.imread(str(orig_path))
    deformed = cv2.imread(str(def_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
    
    h, w = orig.shape[:2]
    
    # Check SHA-256
    h_orig = hashlib.sha256(orig_path.read_bytes()).hexdigest()
    h_def = hashlib.sha256(def_path.read_bytes()).hexdigest()
    
    diff = cv2.absdiff(orig, deformed)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    mod_px = int(np.count_nonzero(diff_gray > 0))
    mod_px_strict = int(np.count_nonzero(diff_gray > 2))
    max_diff = int(diff_gray.max())
    mean_diff = float(diff_gray.mean())
    
    mask_px = int(np.count_nonzero(mask)) if mask is not None else 0
    if mask is not None:
        diff_in_mask = diff_gray[mask > 0]
        mod_in_mask = int(np.count_nonzero(diff_in_mask > 2))
        diff_out_mask = diff_gray[mask == 0]
        mod_out_mask = int(np.count_nonzero(diff_out_mask > 2))
    else:
        mod_in_mask = 0
        mod_out_mask = 0
        
    print(f"\n{v.upper()}:")
    print(f"  Orig: {w}x{h}, SHA256: {h_orig[:12]}...")
    print(f"  Def:  {deformed.shape[1]}x{deformed.shape[0]}, SHA256: {h_def[:12]}...")
    print(f"  Hashes equal: {h_orig == h_def}")
    print(f"  Modified px (>0): {mod_px} ({mod_px/(w*h)*100:.2f}%), (>2): {mod_px_strict} ({mod_px_strict/(w*h)*100:.2f}%)")
    print(f"  Max pixel diff: {max_diff}/255, Mean pixel diff: {mean_diff:.4f}")
    print(f"  Mask pixels: {mask_px}, Mod in mask: {mod_in_mask}, Mod outside mask: {mod_out_mask}")

