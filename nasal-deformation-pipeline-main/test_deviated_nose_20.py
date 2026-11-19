
from pathlib import Path
import cv2

from src.masking import load_nose_mask
from src.deformation import deviated_nose_deformation


ROOT = Path(__file__).resolve().parent
IDENTITY = "H01"

# Start conservatively; tune after visual inspection.

STRENGTH = 0.20


VIEWS = [
    {
        "name": "03_left_40",
        "image": "H01_LEFT40.png",
        "mask_folder": "H01_LEFT40",
        "direction": "left",
    },
    {
        "name": "06_right_40",
        "image": "H01_RIGHT40.png",
        "mask_folder": "H01_RIGHT40",
        "direction": "left",
    },
    {
        "name": "04_left_60",
        "image": "H01_LEFT60.png",
        "mask_folder": "H01_LEFT60",
        "direction": "left",
    },
    {
        "name": "07_right_60",
        "image": "H01_RIGHT60.png",
        "mask_folder": "H01_RIGHT60",
        "direction": "left",
    },
]

IDENTITY_DIR = ROOT / "data" / "identities" / IDENTITY
INPUT_DIR = IDENTITY_DIR / "input"
MASK_DIR = IDENTITY_DIR / "masks"
OUTPUT_DIR = IDENTITY_DIR / "deviated_20"


def process_view(view):
    image_path = INPUT_DIR / view["image"]
    mask_path = MASK_DIR / view["mask_folder"] / "nose_binary.png"

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not mask_path.is_file():
        raise FileNotFoundError(f"Nose mask not found: {mask_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    nose_mask = load_nose_mask(str(mask_path), image.shape)

    result = deviated_nose_deformation(
        image=image,
        nose_mask=nose_mask,
        strength=STRENGTH,
        direction=view["direction"],
    )

    output_dir = OUTPUT_DIR / view["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "original.png": image,
        "nose_mask.png": nose_mask,
        "deviated.png": result,
    }

    for filename, array in outputs.items():
        output_path = output_dir / filename

        if not cv2.imwrite(str(output_path), array):
            raise OSError(f"Could not save: {output_path}")

        print(f"Saved: {output_path}")

    print(
        f"Completed {view['name']} | "
        f"direction={view['direction']} | strength={STRENGTH}"
    )


def main():
    print("DEVIATED NOSE — 20-DEGREE VIEW TEST")
    print(f"Identity: {IDENTITY}")
    print(f"Strength: {STRENGTH}")

    for view in VIEWS:
        process_view(view)

    print("\nBoth 20-degree views processed.")


if __name__ == "__main__":
    main()
