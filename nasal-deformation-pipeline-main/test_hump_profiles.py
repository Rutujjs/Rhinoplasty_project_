
from pathlib import Path

import cv2

from src.masking import load_nose_mask
from src.deformation import hump_nose_deformation


ROOT = Path(__file__).resolve().parent

IDENTITY = "H03"
INPUT_DIR = ROOT / "data" / "identities" / IDENTITY / "input"
MASK_DIR = ROOT / "data" / "identities" / IDENTITY / "masks"
OUTPUT_DIR = ROOT / "data" / "identities" / IDENTITY / "hump"

STRENGTH = 0.55

# Match the filenames you added to the project.
VIEWS = {
    "left20": ("02_left_20.png", "left"),
    "left40": ("03_left_40.png", "left"),
    "left60": ("04_left_60.png", "left"),
    "right20": ("05_right_20.png", "right"),
    "right40": ("06_right_40.png", "right"),
    "right60": ("07_right_60.png", "right"),
}


def main():
    for view, (filename, contour_side) in VIEWS.items():
        image_path = INPUT_DIR / filename
        mask_path = MASK_DIR / Path(filename).stem / "nose_binary.png"

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not mask_path.is_file():
            raise FileNotFoundError(f"Nose mask not found: {mask_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        nose_mask = load_nose_mask(
            str(mask_path),
            image.shape[:2],
        )

        result = hump_nose_deformation(
            image,
            nose_mask,
            strength=STRENGTH,
            contour_side=contour_side,
        )

        output_folder = OUTPUT_DIR / view
        output_folder.mkdir(parents=True, exist_ok=True)

        outputs = {
            "original.png": image,
            "nose_mask.png": nose_mask,
            "hump.png": result,
        }

        for name, output in outputs.items():
            if not cv2.imwrite(str(output_folder / name), output):
                raise OSError(f"Could not save {output_folder / name}")

        print(f"[OK] {view}: {output_folder}")

    print("\nHump profile testing completed.")


if __name__ == "__main__":
    main()
