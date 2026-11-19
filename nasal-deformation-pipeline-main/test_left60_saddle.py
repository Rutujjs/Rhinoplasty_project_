
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image

from src.deformation import left60_saddle_nose


ROOT = Path(__file__).resolve().parent

image_path = ROOT / "data" / "input" / "left60.png"
mask_path = (
    ROOT / "data" / "diagnostics" / "left60_mask_test"
    / "resnet18" / "left60_raw.png"
)
output_dir = ROOT / "data" / "diagnostics" / "left60_saddle_test"
output_dir.mkdir(parents=True, exist_ok=True)

# Load the original image and the indexed BiSeNet mask.
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Cannot read image: {image_path}")

labels = np.array(Image.open(mask_path))
if labels.ndim != 2:
    raise ValueError(f"Expected a 2D label map, got {labels.shape}")

# Extract class 10 (nose), then resize the mask to source dimensions.
nose_mask = (labels == 10).astype(np.uint8) * 255
nose_mask = cv2.resize(
    nose_mask,
    (image.shape[1], image.shape[0]),
    interpolation=cv2.INTER_NEAREST,
)

# Run the first-pass deformation.
# Load the detected facial landmarks
landmark_path = (
    ROOT / "data" / "diagnostics" /
    "landmarks_left60" / "left60_landmarks.json"
)

landmark_data = json.loads(
    landmark_path.read_text(encoding="utf-8")
)

landmarks = {
    point["id"]: (point["x"], point["y"])
    for point in landmark_data["landmarks"]
}

# Apply the left60 saddle-nose deformation
result = left60_saddle_nose(
    image=image,
    nose_mask=nose_mask,
    strength=0.65,
)
difference = cv2.absdiff(image, result)
changed_pixels = np.any(difference > 5, axis=2)

print("Pixels changed significantly:", int(changed_pixels.sum()))
print("Maximum channel difference:", int(difference.max()))

cv2.imwrite(
    str(output_dir / "left60_difference.png"),
    cv2.applyColorMap(
        cv2.normalize(
            difference.max(axis=2),
            None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8),
        cv2.COLORMAP_JET,
    ),
)

# Save the result and a side-by-side comparison.
cv2.imwrite(str(output_dir / "left60_saddle.png"), result)
cv2.imwrite(str(output_dir / "left60_original.png"), image)

comparison = np.hstack((image, result))
cv2.imwrite(str(output_dir / "left60_comparison.png"), comparison)

print("Original dimensions:", image.shape)
print("Nose-mask pixels after resize:", int(np.count_nonzero(nose_mask)))
print("Saved outputs to:", output_dir.resolve())
