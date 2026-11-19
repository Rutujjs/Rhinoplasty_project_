
from pathlib import Path
import cv2

from src.masking import load_nose_mask
from src.deformation import helicopter_broad_nose

ROOT = Path(__file__).resolve().parent

IMAGE_PATH = ROOT / "data" / "input" / "helicopter_view.png"
MASK_PATH = ROOT / "data" / "masks" / "helicopter_view" / "nose_binary.png"
OUTPUT_DIR = ROOT / "data" / "broad_validation" / "helicopter_view"


def main():
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    nose_mask = load_nose_mask(
        str(MASK_PATH),
        image.shape[:2]
    )

    result = helicopter_broad_nose(
        image,
        nose_mask,
        strength=0.25
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(OUTPUT_DIR / "original.png"), image)
    cv2.imwrite(str(OUTPUT_DIR / "nose_mask.png"), nose_mask)
    cv2.imwrite(str(OUTPUT_DIR / "broad_nose.png"), result)

    print(f"Saved helicopter broad-nose test to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
