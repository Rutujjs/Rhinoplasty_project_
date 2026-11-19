import hashlib
from pathlib import Path
import cv2
import numpy as np

views = ["frontal", "basal", "helicopter", "left_20", "left_40", "left_60", "right_20", "right_40", "right_60"]

orig_dir = Path("storage/generated/P01/original")
def_dir = Path("storage/deformations/7/P01/broad")

print(f"{'View':<12} | {'Orig Exist':<10} | {'Def Exist':<10} | {'Orig SHA':<12} | {'Def SHA':<12} | {'Identical?':<10} | {'Max Diff':<8} | {'Mean Diff':<10} | {'Changed Pct':<10}")
print("-" * 105)

for v in views:
    orig_path = orig_dir / f"{v}.png"
    def_path = def_dir / f"{v}_broad.png"
    mask_path = def_dir / f"{v}_mask.png"
    
    orig_exists = orig_path.exists()
    def_exists = def_path.exists()
    
    if not orig_exists or not def_exists:
        print(f"{v:<12} | {str(orig_exists):<10} | {str(def_exists):<10}")
        continue
        
    orig_bytes = orig_path.read_bytes()
    def_bytes = def_path.read_bytes()
    
    orig_sha = hashlib.sha256(orig_bytes).hexdigest()[:10]
    def_sha = hashlib.sha256(def_bytes).hexdigest()[:10]
    is_sha_same = (orig_sha == def_sha)
    
    orig_img = cv2.imread(str(orig_path))
    def_img = cv2.imread(str(def_path))
    
    if orig_img.shape != def_img.shape:
        print(f"{v:<12} | Shape mismatch: {orig_img.shape} vs {def_img.shape}")
        continue
        
    diff = cv2.absdiff(orig_img, def_img)
    max_d = np.max(diff)
    mean_d = np.mean(diff)
    changed_pixels = np.count_nonzero(np.any(diff > 0, axis=-1))
    total_pixels = orig_img.shape[0] * orig_img.shape[1]
    pct = (changed_pixels / total_pixels) * 100
    
    print(f"{v:<12} | {str(orig_exists):<10} | {str(def_exists):<10} | {orig_sha:<12} | {def_sha:<12} | {str(is_sha_same):<10} | {max_d:<8} | {mean_d:<10.4f} | {pct:<10.2f}%")

