
from pathlib import Path

import cv2

from src.masking import load_nose_mask
from src.deformation import helicopter_hump_nose


ROOT = Path(__file__).resolve().parent

IDENTITY = "H03"
INPUT_DIR = ROOT / "data" / "identities" / IDENTITY / "input"
MASK_DIR = ROOT / "data" / "identities" / IDENTITY / "masks"
OUTPUT_DIR = ROOT / "data" / "identities" / IDENTITY / "hump"

STRENGTH = 0.25

VIEWS = {
    "helicopter": "09_helicopter_view.png",
}


def main():
    for view, filename in VIEWS.items():
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

        result = helicopter_hump_nose(
            image,
            nose_mask,
            strength=STRENGTH,
        )

        output_folder = OUTPUT_DIR / view
        output_folder.mkdir(parents=True, exist_ok=True)

        outputs = {
            "original.png": image,
            "nose_mask.png": nose_mask,
            "hump.png": result,
        }

        for name, output in outputs.items():
            output_path = output_folder / name

            if not cv2.imwrite(str(output_path), output):
                raise OSError(f"Could not save {output_path}")

        print(f"[OK] {view}: {output_folder}")

    print("\nSpecial-view hump test completed.")


if __name__ == "__main__":
    main()
