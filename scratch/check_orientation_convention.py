import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ref_input_dir = BASE_DIR / "nasal-deformation-pipeline-main" / "data" / "input"
p01_dir = BASE_DIR / "storage" / "generated" / "P01" / "original"

views = [
    ("left_20", "left20.png"),
    ("left_40", "left40.png"),
    ("left_60", "left60.png"),
    ("right_20", "right20.png"),
    ("right_40", "right40.png"),
    ("right_60", "right60.png"),
]

print("=== CHECKING ORIENTATION CONVENTION (REF vs P01) ===")
for v, ref_name in views:
    ref_img = cv2.imread(str(ref_input_dir / ref_name))
    p01_img = cv2.imread(str(p01_dir / f"{v}.png"))
    
    # In ref_img, where is the face / nose?
    # Let's find the horizontal center of mass of non-background or edges
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    p01_gray = cv2.cvtColor(p01_img, cv2.COLOR_BGR2GRAY)
    
    # Check edges
    ref_edges = cv2.Canny(ref_gray, 50, 150)
    p01_edges = cv2.Canny(p01_gray, 50, 150)
    
    ref_xs = np.where(ref_edges > 0)[1]
    p01_xs = np.where(p01_edges > 0)[1]
    
    print(f"\n{v}:")
    print(f"  Ref  image edge mean x: {ref_xs.mean():.1f} / {ref_img.shape[1]} (rel: {ref_xs.mean()/ref_img.shape[1]:.2f})")
    print(f"  P01 image edge mean x: {p01_xs.mean():.1f} / {p01_img.shape[1]} (rel: {p01_xs.mean()/p01_img.shape[1]:.2f})")
    
    # Also check the mask
    mask_file = BASE_DIR / "deformation" / "reference_masks" / v.replace("_", "") / "nose_binary.png"
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    mask_xs = np.where(mask > 0)[1]
    print(f"  Mask active mean x:    {mask_xs.mean():.1f} / {mask.shape[1]} (rel: {mask_xs.mean()/mask.shape[1]:.2f})")

