import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
p01_dir = BASE_DIR / "storage" / "generated" / "P01" / "original"

# In left_60 and right_60, let's find the extreme horizontal silhouette points across y = 150..250 (midface)
for v in ["left_60", "right_60"]:
    img = cv2.imread(str(p01_dir / f"{v}.png"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Background in synthetic images is typically uniform or light/dark
    # Let's inspect rows around midface (y = 180 to 220)
    mid_rows = gray[180:220, :]
    # Let's print min and max non-background x coordinates
    # Threshold from corner background pixel
    bg_val = int(gray[10, 10])
    diff_from_bg = np.abs(gray.astype(int) - bg_val)
    fg_mask = (diff_from_bg > 25).astype(np.uint8)
    
    fg_in_mid = fg_mask[180:220, :]
    ys, xs = np.where(fg_in_mid > 0)
    if len(xs) > 0:
        print(f"{v}: Foreground x range: {xs.min()}..{xs.max()} (w={w}), center={xs.mean():.1f}")

