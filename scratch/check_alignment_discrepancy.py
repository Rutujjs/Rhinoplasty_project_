import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ref_dir = BASE_DIR / "deformation" / "reference_masks"
p01_dir = BASE_DIR / "storage" / "generated" / "P01" / "original"
ref_input_dir = BASE_DIR / "nasal-deformation-pipeline-main" / "data" / "input"

views = [
    ("frontal", "frontal_0", "frontal_0.png"),
    ("left_20", "left20", "left20.png"),
    ("left_40", "left40", "left40.png"),
    ("left_60", "left60", "left60.png"),
    ("right_20", "right20", "right20.png"),
    ("right_40", "right40", "right40.png"),
    ("right_60", "right60", "right60.png"),
    ("basal", "basal_view", "basal_view.jpeg"),
    ("helicopter", "helicopter_view", "helicopter_view.png"),
]

print("=== CHECKING MASK ALIGNMENT ROOT CAUSE ===")
for v, ref_sub, ref_img_name in views:
    # 1. Reference mask
    mask_file = ref_dir / ref_sub / "nose_binary.png"
    ref_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    
    # 2. Reference template image (the one from which the mask was created)
    ref_img_file = ref_input_dir / ref_img_name
    ref_img = cv2.imread(str(ref_img_file)) if ref_img_file.exists() else None
    
    # 3. P01 image
    p01_file = p01_dir / f"{v}.png"
    p01_img = cv2.imread(str(p01_file)) if p01_file.exists() else None
    
    print(f"\nView {v.upper()}:")
    print(f"  Ref input: {ref_img.shape if ref_img is not None else 'None'}")
    print(f"  P01 image: {p01_img.shape if p01_img is not None else 'None'}")
    print(f"  Ref mask:  {ref_mask.shape if ref_mask is not None else 'None'}")
    
    if p01_img is not None and ref_mask is not None:
        h_p, w_p = p01_img.shape[:2]
        # Resized mask
        resized_mask = cv2.resize(ref_mask, (w_p, h_p), interpolation=cv2.INTER_NEAREST)
        
        # In P01, what is the average color/luminance under the mask?
        p01_nose_crop = p01_img[resized_mask > 0]
        p01_mean_bgr = p01_nose_crop.mean(axis=0) if len(p01_nose_crop) > 0 else [0,0,0]
        print(f"  P01 mask area: {np.count_nonzero(resized_mask)}px, mean BGR under mask: [{p01_mean_bgr[0]:.1f}, {p01_mean_bgr[1]:.1f}, {p01_mean_bgr[2]:.1f}]")

