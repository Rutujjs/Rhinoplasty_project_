
from pathlib import Path
import argparse

import cv2

from src.masking import load_nose_mask
from src.deformation import (
    saddle_nose_deformation,
    frontal_saddle_nose,
    basal_saddle_nose,
    helicopter_saddle_nose,
)


ROOT = Path(__file__).resolve().parent
STRENGTH = 0.65

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Each key is the logical view name.
# The value identifies the corresponding image filename.
VIEW_TOKENS = {
    "frontal": "FRONTAL",
    "left20": "LEFT20",
    "left40": "LEFT40",
    "left60": "LEFT60",
    "right20": "RIGHT20",
    "right40": "RIGHT40",
    "right60": "RIGHT60",
    "basal_view": "BASAL",
    "helicopter_view": "HELICOPTER",
}


def find_view_image(input_dir, token):
    """Find exactly one original image matching a view token."""

    candidates = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and token.lower() in path.stem.lower()
    ]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No image matching '{token}' was found in {input_dir}"
        )

    raise RuntimeError(
        f"Multiple images match '{token}': "
        f"{[path.name for path in candidates]}. "
        "Keep only one matching original image or rename the files."
    )


def deform_view(view, image, nose_mask):
    """Call the existing approved deformation function for this view."""

    if view == "frontal":
        return frontal_saddle_nose(
            image, nose_mask, strength=STRENGTH
        )

    if view.startswith("left"):
        return saddle_nose_deformation(
            image,
            nose_mask,
            strength=STRENGTH,
            contour_side="left",
        )

    if view.startswith("right"):
        return saddle_nose_deformation(
            image,
            nose_mask,
            strength=STRENGTH,
            contour_side="right",
        )

    if view == "basal_view":
        return basal_saddle_nose(
            image, nose_mask, strength=STRENGTH
        )

    if view == "helicopter_view":
        return helicopter_saddle_nose(
            image, nose_mask, strength=STRENGTH
        )

    raise ValueError(f"Unsupported view: {view}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate saddle-nose results for all views of one identity."
    )
    parser.add_argument(
        "--identity",
        required=True,
        help="Identity folder name, for example H01",
    )
    args = parser.parse_args()

    identity_dir = ROOT / "data" / "identities" / args.identity
    input_dir = identity_dir / "input"
    mask_dir = identity_dir / "masks"
    output_dir = identity_dir / "saddle"

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Identity input folder not found: {input_dir}"
        )

    print(f"Identity: {args.identity}")
    print(f"Input:    {input_dir}")
    print(f"Masks:    {mask_dir}")
    print(f"Output:   {output_dir}")
    print(f"Strength: {STRENGTH}")
    print("-" * 55)

    for view, token in VIEW_TOKENS.items():
        image_path = find_view_image(input_dir, token)

        # Mask folders are named after the original image stem,
        # matching the output structure of the identity-aware mask script.
        mask_path = mask_dir / image_path.stem / "nose_binary.png"

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Nose mask not found for {image_path.name}: {mask_path}"
            )

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(
                f"Could not read original image: {image_path}"
            )

        nose_mask = load_nose_mask(
            str(mask_path),
            image.shape[:2],
        )

        result = deform_view(view, image, nose_mask)

        view_output_dir = output_dir / view
        view_output_dir.mkdir(parents=True, exist_ok=True)

        outputs = (
            ("original.png", image),
            ("nose_mask.png", nose_mask),
            ("saddle.png", result),
        )

        for filename, output in outputs:
            saved = cv2.imwrite(
                str(view_output_dir / filename),
                output,
            )
            if not saved:
                raise OSError(
                    f"Could not save {filename} for {view_output_dir}"
                )

        print(f"[OK] {view}: {view_output_dir}")

    print("-" * 55)
    print(f"Finished processing identity {args.identity}.")


if __name__ == "__main__":
    main()
