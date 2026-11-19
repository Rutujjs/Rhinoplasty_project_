
from pathlib import Path
import subprocess
import sys

import cv2

from src.masking import load_nose_mask
from src.deformation import deviated_nose_deformation


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
IDENTITY = "H01"
VIEW = "H01_FRONTAL.png"
STRENGTH = 0.20
DIRECTION = "right"

IDENTITY_DIR = ROOT / "data" / "identities" / IDENTITY
INPUT_DIR = IDENTITY_DIR / "input"
MASK_DIR = IDENTITY_DIR / "masks"
OUTPUT_DIR = IDENTITY_DIR / "deviated"

IMAGE_PATH = INPUT_DIR / VIEW
VIEW_NAME = Path(VIEW).stem
MASK_PATH = MASK_DIR / VIEW_NAME / "nose_binary.png"


def generate_masks():
    """Run the existing mask-generation pipeline for this identity."""

    script_path = ROOT / "generate_masks.py"

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Mask-generation script not found: {script_path}"
        )

    print(f"\n[1/4] Generating nose masks for {IDENTITY}...")

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--identity",
            IDENTITY,
        ],
        cwd=str(ROOT),
        check=True,
    )

    print("Mask generation completed.")


def load_inputs():
    """Load the original frontal image and generated nose mask."""

    print("\n[2/4] Loading image and nose mask...")

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

    if not MASK_PATH.is_file():
        raise FileNotFoundError(
            f"Generated nose mask not found: {MASK_PATH}"
        )

    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {IMAGE_PATH}")

    nose_mask = load_nose_mask(str(MASK_PATH), image.shape)

    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
    print("Image and nose mask loaded successfully.")

    return image, nose_mask


def apply_deformation(image, nose_mask):
    """Call the reusable function from src/deformation.py."""

    print("\n[3/4] Applying deviated-nose deformation...")

    result = deviated_nose_deformation(
        image=image,
        nose_mask=nose_mask,
        strength=STRENGTH,
        direction=DIRECTION,
    )

    return result


def save_results(image, nose_mask, result):
    """Save the original, mask and deformed image."""

    print("\n[4/4] Saving results...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "original.png": image,
        "nose_mask.png": nose_mask,
        "deviated_frontal.png": result,
    }

    for filename, array in outputs.items():
        output_path = OUTPUT_DIR / filename

        if not cv2.imwrite(str(output_path), array):
            raise OSError(f"Could not save output: {output_path}")

        print(f"Saved: {output_path}")

    print("\nWorkflow completed successfully!")


def main():
    print("=" * 55)
    print("DEVIATED NOSE DEFORMATION TEST")
    print(f"Identity: {IDENTITY}")
    print(f"View: {VIEW}")
    print(f"Strength: {STRENGTH}")
    print(f"Direction: {DIRECTION}")
    print("=" * 55)

    # Step 1: Generate masks before loading the mask.
    generate_masks()

    # Step 2: Load original image and generated mask.
    image, nose_mask = load_inputs()

    # Step 3: Call the shared deformation implementation.
    result = apply_deformation(image, nose_mask)

    # Step 4: Save outputs.
    save_results(image, nose_mask, result)


if __name__ == "__main__":
    main()
