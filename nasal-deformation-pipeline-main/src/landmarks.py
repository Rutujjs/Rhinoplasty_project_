
from pathlib import Path
import json

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from src.deformation import left60_saddle_nose


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mediapipe" / "face_landmarker.task"

IMAGE_PATH = ROOT / "data" / "input" / "left60.png"
MASK_PATH = (
    ROOT / "data" / "diagnostics" / "left60_mask_test"
    / "resnet18" / "left60_raw.png"
)
OUTPUT_DIR = ROOT / "data" / "diagnostics" / "left60_saddle_test"


def detect_face_landmarks(image_path):
    image_path = Path(image_path)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Landmarker model not found: {MODEL_PATH}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )

    if not result.face_landmarks:
        raise RuntimeError("No face detected in the image.")

    points = [
        {
            "id": i,
            "x": float(p.x * width),
            "y": float(p.y * height),
            "z": float(p.z),
        }
        for i, p in enumerate(result.face_landmarks[0])
    ]

    return points


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if not MASK_PATH.exists():
        raise FileNotFoundError(f"BiSeNet mask not found: {MASK_PATH}")

    # Load the indexed BiSeNet mask and select nose class 10.
    labels = np.array(Image.open(MASK_PATH))
    if labels.ndim != 2:
        raise ValueError("Expected a single-channel indexed BiSeNet mask.")

    nose_mask = (labels == 10).astype(np.uint8) * 255
    nose_mask = cv2.resize(
        nose_mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # Detect and save landmarks for diagnostics.
    landmarks = detect_face_landmarks(IMAGE_PATH)

    landmark_path = OUTPUT_DIR / "left60_landmarks.json"
    landmark_path.write_text(
        json.dumps({"landmarks": landmarks}, indent=2),
        encoding="utf-8",
    )

    preview = image.copy()
    for p in landmarks:
        x, y = int(round(p["x"])), int(round(p["y"]))
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(preview, (x, y), 1, (0, 255, 0), -1)

    cv2.imwrite(str(OUTPUT_DIR / "left60_landmarks.png"), preview)

    # Apply saddle deformation using the BiSeNet nose mask.
    # This function signature does not require landmarks.
    result = left60_saddle_nose(
        image,
        nose_mask,
        strength=0.65,
    )

    cv2.imwrite(str(OUTPUT_DIR / "left60_saddle_result.png"), result)
    cv2.imwrite(str(OUTPUT_DIR / "left60_original.png"), image)

    difference = cv2.absdiff(image, result)
    cv2.imwrite(str(OUTPUT_DIR / "left60_difference.png"), difference)

    print("Landmarks saved:", landmark_path)
    print("Saddle result:", OUTPUT_DIR / "left60_saddle_result.png")
    print("Difference image:", OUTPUT_DIR / "left60_difference.png")


if __name__ == "__main__":
    main()
