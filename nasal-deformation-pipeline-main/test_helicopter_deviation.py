
import cv2
from pathlib import Path

from src.deformation import helicopter_deviated_nose_deformation


ROOT = Path(__file__).resolve().parent

IMAGE_PATH = (
    ROOT / "data/identities/H01/input/H01_HELICOPTER.png"
)
MASK_PATH = (
    ROOT / "data/identities/H03/masks/09_helicopter_view/nose_binary.png"
)
OUTPUT_DIR = ROOT / "data/identities/H03/deviated_helicopter"

STRENGTH = 0.35
DIRECTION = "right"


def main():
    image = cv2.imread(str(IMAGE_PATH))
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {MASK_PATH}")

    result = helicopter_deviated_nose_deformation(
        image,
        mask,
        strength=STRENGTH,
        direction=DIRECTION,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(OUTPUT_DIR / "original.png"), image)
    cv2.imwrite(str(OUTPUT_DIR / "nose_mask.png"), mask)
    cv2.imwrite(str(OUTPUT_DIR / "deviated_helicopter.png"), result)

    print(f"Original: {IMAGE_PATH}")
    print(f"Mask: {MASK_PATH}")
    print(f"Output: {OUTPUT_DIR / 'deviated_helicopter.png'}")
    print(f"Strength: {STRENGTH}, direction: {DIRECTION}")


if __name__ == "__main__":
    main()
