import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
p01_dir = BASE_DIR / "storage" / "generated" / "P01" / "original"

frontal = cv2.imread(str(p01_dir / "frontal.png"))
h, w = frontal.shape[:2]

# Let's crop patches from y=100 to y=250 around center x
# y=100-140: eyes/bridge?
# y=140-190: nose dorsum / tip?
# y=190-240: tip / columella / lips?
# Let's inspect mean brightness and standard deviation
for y in range(100, 260, 20):
    patch = frontal[y:y+20, w//2-25:w//2+25]
    print(f"y={y:>3}..{y+20:>3}: BGR mean=[{patch[:,:,0].mean():.1f}, {patch[:,:,1].mean():.1f}, {patch[:,:,2].mean():.1f}], std={patch.std():.1f}")

