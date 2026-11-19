
from pathlib import Path

import cv2

from src.masking import load_nose_mask
from src.deformation import (
    basal_saddle_nose,
    helicopter_saddle_nose,
)

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "input"
MASK_DIR = ROOT / "data" / "masks"
OUTPUT_DIR = ROOT / "data" / "saddle_validation"

VIEWS = {
    "basal_view": ("basal_view.jpeg", basal_saddle_nose),
    "helicopter_view": ("helicopter_view.png", helicopter_saddle_nose),
}


def main():
    for view, (filename, deformation_function) in VIEWS.items():
        image_path = INPUT_DIR / filename
        mask_path = MASK_DIR / view / "nose_binary.png"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        nose_mask = load_nose_mask(str(mask_path), image.shape[:2])

        result = deformation_function(
            image,
            nose_mask,
            strength=0.65,
        )

        output_folder = OUTPUT_DIR / view
        output_folder.mkdir(parents=True, exist_ok=True)

        for name, output in (
            ("original.png", image),
            ("nose_mask.png", nose_mask),
            ("saddle.png", result),
        ):
            if not cv2.imwrite(str(output_folder / name), output):
                raise OSError(f"Could not save {name} for {view}")

        print(f"Saved {view} results to {output_folder}")


if __name__ == "__main__":
    main()
