
from pathlib import Path

import cv2

from src.masking import load_nose_mask
from src.deformation import saddle_nose_deformation


ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "data" / "input"
MASK_DIR = ROOT / "data" / "masks"
OUTPUT_DIR = ROOT / "data" / "saddle_validation"

# Updated to validate the three right-facing profile views.
VIEWS = {
    "right20": "right",
    "right40": "right",
    "right60": "right",
}

STRENGTH = 0.65


def find_image(view):
    folder = INPUT_DIR / view

    if folder.is_dir():
        candidates = [
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and p.stem.lower() not in {
                "labels",
                "nose_binary",
                "nose_overlay",
            }
        ]

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple images found in {folder}; "
                "select the original image explicitly."
            )

    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = INPUT_DIR / f"{view}{suffix}"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Original image not found for {view} in {INPUT_DIR}"
    )


def main():
    for view, contour_side in VIEWS.items():
        image_path = find_image(view)
        mask_path = MASK_DIR / view / "nose_binary.png"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(
                f"Could not read original image: {image_path}"
            )

        nose_mask = load_nose_mask(
            str(mask_path),
            image.shape[:2],
        )

        result = saddle_nose_deformation(
            image,
            nose_mask,
            strength=STRENGTH,
            contour_side=contour_side,
        )

        output_folder = OUTPUT_DIR / view
        output_folder.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(
            str(output_folder / "original.png"), image
        ):
            raise OSError(f"Could not save original image for {view}")

        if not cv2.imwrite(
            str(output_folder / "nose_mask.png"), nose_mask
        ):
            raise OSError(f"Could not save mask for {view}")

        if not cv2.imwrite(
            str(output_folder / "saddle.png"), result
        ):
            raise OSError(f"Could not save saddle result for {view}")

        print(f"Saved saddle validation for {view}: {output_folder}")


if __name__ == "__main__":
    main()
