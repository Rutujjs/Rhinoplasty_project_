import cv2
import numpy as np

p1 = "storage/diagnostics/bisenet_patient_masks/bisenet_mask_helicopter.png"
p2 = "storage/deformations/9/P01/deviated/helicopter_mask.png"

m1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
m2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

print("m1:", m1.shape if m1 is not None else None, np.count_nonzero(m1) if m1 is not None else 0)
print("m2:", m2.shape if m2 is not None else None, np.count_nonzero(m2) if m2 is not None else 0)

