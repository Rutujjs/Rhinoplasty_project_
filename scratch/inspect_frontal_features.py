import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
p01_dir = BASE_DIR / "storage" / "generated" / "P01" / "original"

frontal = cv2.imread(str(p01_dir / "frontal.png"))
h, w = frontal.shape[:2]

# Let's crop vertical slices across center x = 192 (w=384)
# Let's print the colors along the vertical center line x=192 from y=80 to y=300
print(f"Frontal image shape: {w}x{h}")

# The nose tip usually has highlights (lighter skin tone) or nostril shadows (darker)
gray = cv2.cvtColor(frontal, cv2.COLOR_BGR2GRAY)
center_col = gray[:, w//2]

# Also check mask overlay in diagnostic_frontal.png
mask = cv2.imread(str(BASE_DIR / "storage" / "diagnostics" / "mask_audit" / "mask_frontal.png"), cv2.IMREAD_GRAYSCALE)
ys, xs = np.where(mask > 0)
print(f"Frontal mask y range: {ys.min()}..{ys.max()} (center y={ys.mean():.1f}), x range: {xs.min()}..{xs.max()} (center x={xs.mean():.1f})")

# Let's check gradient in gray
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
mag = np.sqrt(sobel_x**2 + sobel_y**2)

# Along center x +/- 20, where are the strong horizontal edges? (eyes, nose base, lips)
profile = np.mean(mag[:, w//2-20:w//2+20], axis=1)
peaks = [y for y in range(50, 320) if profile[y] > np.percentile(profile, 85) and profile[y] == max(profile[max(0, y-5):min(len(profile), y+6)])]
print(f"Prominent horizontal facial features (y-coordinates): {peaks}")

