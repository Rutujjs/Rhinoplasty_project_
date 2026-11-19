from pathlib import Path
import cv2
import numpy as np

# Load original FLUX image
img = cv2.imread("storage/generated/P01/original/frontal.png")
mask = cv2.imread("storage/deformations/7/P01/broad/frontal_mask.png", cv2.IMREAD_GRAYSCALE)

print(f"Image shape: {img.shape}")
print(f"Mask shape: {mask.shape}")

# Let's check image statistics: brightness, center of image
h, w, c = img.shape

# Let's save a grid of the original image with coordinates and mask overlay
overlay = img.copy()

# Draw grid lines every 50 pixels
for x in range(0, w, 50):
    cv2.line(overlay, (x, 0), (x, h), (100, 100, 100), 1)
for y in range(0, h, 50):
    cv2.line(overlay, (0, y), (w, y), (100, 100, 100), 1)

# Color mask region in green
overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

# Find mask bounding box
ys, xs = np.where(mask > 0)
cv2.rectangle(overlay, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 0, 255), 2)
cv2.putText(overlay, f"Mask: [{xs.min()},{ys.min()}]-[{xs.max()},{ys.max()}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

out_path = Path("storage/deformations/audit_diagnostics/diagnostic_frontal_mask_check.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_path), overlay)
print(f"Saved diagnostic overlay to {out_path}")

