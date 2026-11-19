from pathlib import Path
import cv2
import numpy as np

img = cv2.imread("storage/generated/P01/original/frontal.png")
print("Image shape:", img.shape)
print("Min, max, mean pixel values:", img.min(), img.max(), img.mean(axis=(0,1)))

# Crop the region of the mask [131:190, 171:221]
mask = cv2.imread("storage/deformations/7/P01/broad/frontal_mask.png", cv2.IMREAD_GRAYSCALE)
ys, xs = np.where(mask > 0)
crop = img[ys.min():ys.max()+1, xs.min():xs.max()+1]
print("Crop shape:", crop.shape)
print("Crop mean color (BGR):", crop.mean(axis=(0,1)))

# Let's inspect the entire image: is it a realistic face photo or something else?
# Let's check color channels in different parts of the image (top, center, bottom, background corners)
print("Top-left corner (0:30, 0:30) mean BGR:", img[0:30, 0:30].mean(axis=(0,1)))
print("Center (150:200, 150:200) mean BGR:", img[150:200, 150:200].mean(axis=(0,1)))
print("Bottom center (300:350, 150:200) mean BGR:", img[300:350, 150:200].mean(axis=(0,1)))

