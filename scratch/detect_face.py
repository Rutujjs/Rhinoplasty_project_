from pathlib import Path
import cv2
import numpy as np

img = cv2.imread("storage/generated/P01/original/frontal.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.py' if hasattr(cv2.data, 'haarcascades') else '')
# Try loading standard Haar cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
print("Cascade exists:", Path(cascade_path).exists())
face_cascade = cv2.CascadeClassifier(cascade_path)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print("Detected faces:", faces)

# Also check nose cascade
nose_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_nose.xml'
print("Nose cascade exists:", Path(nose_cascade_path).exists())

# Let's save a visualization of the mask on the image
mask = cv2.imread("storage/deformations/7/P01/broad/frontal_mask.png", cv2.IMREAD_GRAYSCALE)
overlay = img.copy()
overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

# Draw bounding box of mask in green
ys, xs = np.where(mask > 0)
cv2.rectangle(overlay, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 0), 2)

for (fx, fy, fw, fh) in faces:
    cv2.rectangle(overlay, (fx, fy), (fx+fw, fy+fh), (255, 255, 0), 2)

cv2.imwrite("storage/deformations/audit_diagnostics/mask_overlay_frontal.png", overlay)
print("Saved mask overlay to storage/deformations/audit_diagnostics/mask_overlay_frontal.png")

