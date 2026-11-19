
from pathlib import Path

import cv2

from src.masking import load_nose_mask
from src.deformation import frontal_saddle_nose


ROOT = Path(__file__).resolve().parent

IMAGE_PATH = ROOT / "data" / "input" / "frontal_0.png"
MASK_PATH = ROOT / "data" / "masks" / "frontal_0" / "nose_binary.png"
OUTPUT_DIR = ROOT / "data" / "saddle_validation" / "frontal_0"


def main():
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    nose_mask = load_nose_mask(
        str(MASK_PATH),
        image.shape[:2],
    )

    result = frontal_saddle_nose(
        image,
        nose_mask,
        strength=0.65,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(OUTPUT_DIR / "original.png"), image)
    cv2.imwrite(str(OUTPUT_DIR / "nose_mask.png"), nose_mask)
    cv2.imwrite(str(OUTPUT_DIR / "saddle.png"), result)

    print(f"Saved frontal saddle test to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
