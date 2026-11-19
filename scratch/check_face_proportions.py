import sys
sys.path.insert(0, ".")
from pathlib import Path
import cv2
import numpy as np

img = cv2.imread("storage/generated/P01/original/frontal.png")
mask = cv2.imread("storage/deformations/7/P01/broad/frontal_mask.png", cv2.IMREAD_GRAYSCALE)
h, w = img.shape[:2]

# Compute Sobel edge magnitude
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
mag = cv2.magnitude(gx, gy)

# Let's find vertical profile of edges (horizontal projection of edges)
vert_edges = np.mean(mag, axis=1)
# In human faces:
# Peak 1: Hairline / eyebrows
# Peak 2: Eyes
# Peak 3: Nose tip / nostrils
# Peak 4: Mouth / lips
# Peak 5: Chin
peaks = []
for y in range(10, h-10):
    if vert_edges[y] > vert_edges[y-5:y+6].max() * 0.99 and vert_edges[y] > np.mean(vert_edges):
        peaks.append((y, vert_edges[y]))

print("Edge peaks along Y axis:")
for y, v in peaks:
    print(f"  y={y}: edge magnitude={v:.1f}")

ys, xs = np.where(mask > 0)
print(f"Mask Y range: [{ys.min()}, {ys.max()}], center Y = {np.mean(ys):.1f}")
print(f"Mask X range: [{xs.min()}, {xs.max()}], center X = {np.mean(xs):.1f}")
print(f"Image center: Y={h/2}, X={w/2}")

