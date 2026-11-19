import cv2
import numpy as np
from pathlib import Path
import onnxruntime as ort

model_path = Path("nasal-deformation-pipeline-main/models/bisenet/weights/resnet18.onnx")
img_path = Path("storage/generated/P01/original/frontal.png")

session = ort.InferenceSession(str(model_path))

# Load image
img_bgr = cv2.imread(str(img_path))
h_orig, w_orig = img_bgr.shape[:2]
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Resize to 512x512
img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

# Preprocessing for BiSeNet CelebAMask-HQ:
# Normalize with ImageNet mean/std: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
norm_img = (img_resized.astype(np.float32) / 255.0 - mean) / std

# Transpose to NCHW
tensor = np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

# Run inference
outputs = session.run(None, {"input": tensor})
output_logits = outputs[0]  # shape: (1, 19, 512, 512)
print("Output logits shape:", output_logits.shape)

# Argmax along class dimension (axis 1)
pred_labels_512 = np.argmax(output_logits[0], axis=0).astype(np.uint8)
print("Predicted unique classes:", np.unique(pred_labels_512))

# Check class 10 (nose)
nose_mask_512 = (pred_labels_512 == 10).astype(np.uint8) * 255
print("Nose pixels in 512x512:", np.count_nonzero(nose_mask_512))

# Resize mask back to original dimensions
nose_mask_orig = cv2.resize(nose_mask_512, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
print(f"Original image shape: {w_orig}x{h_orig}, Nose pixels in orig: {np.count_nonzero(nose_mask_orig)}")

ys, xs = np.where(nose_mask_orig > 0)
print(f"Predicted nose bbox: x=[{xs.min()}..{xs.max()}], y=[{ys.min()}..{ys.max()}], center=({xs.mean():.1f}, {ys.mean():.1f})")

# Save inspection overlay
overlay = img_bgr.copy()
red = np.zeros_like(img_bgr)
red[:, :] = (0, 0, 255)
selected = nose_mask_orig > 0
overlay[selected] = (0.45 * img_bgr[selected].astype(np.float32) + 0.55 * red[selected].astype(np.float32)).astype(np.uint8)
cv2.imwrite("scratch/bisenet_frontal_predicted_overlay.png", overlay)
cv2.imwrite("scratch/bisenet_frontal_predicted_mask.png", nose_mask_orig)
print("Saved scratch/bisenet_frontal_predicted_overlay.png")

