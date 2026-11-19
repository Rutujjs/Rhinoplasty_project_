
from pathlib import Path
import cv2

from src.masking import load_nose_mask
from src.deformation import basal_deviated_nose_deformation


ROOT = Path(__file__).resolve().parent
IDENTITY = "H01"
VIEW = "H01_BASAL.png"

STRENGTH = 0.25
DIRECTION = "left"

IDENTITY_DIR = ROOT / "data" / "identities" / IDENTITY
IMAGE_PATH = IDENTITY_DIR / "input" / VIEW
MASK_PATH = (
    IDENTITY_DIR / "masks" / Path(VIEW).stem / "nose_binary.png"
)
OUTPUT_DIR = IDENTITY_DIR / "deviated_basal"


def main():
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if not MASK_PATH.is_file():
        raise FileNotFoundError(f"Mask not found: {MASK_PATH}")

    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {IMAGE_PATH}")

    nose_mask = load_nose_mask(str(MASK_PATH), image.shape)

    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
    print(f"Strength: {STRENGTH}")
    print(f"Direction: {DIRECTION}")

    result = basal_deviated_nose_deformation(
        image=image,
        nose_mask=nose_mask,
        strength=STRENGTH,
        direction=DIRECTION,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "original.png": image,
        "nose_mask.png": nose_mask,
        "deviated_basal.png": result,
    }

    for filename, array in outputs.items():
        output_path = OUTPUT_DIR / filename

        if not cv2.imwrite(str(output_path), array):
            raise OSError(f"Could not save: {output_path}")

        print(f"Saved: {output_path}")

    print("Basal deformation test completed.")


if __name__ == "__main__":
    main()
