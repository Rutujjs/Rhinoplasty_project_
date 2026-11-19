
from pathlib import Path
import cv2

from src.masking import load_nose_mask
from src.deformation import frontal_broad_nose, profile_broad_nose


ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "data" / "input"
MASK_DIR = ROOT / "data" / "masks"
OUTPUT_DIR = ROOT / "data" / "broad_validation"

# Each image is processed using its corresponding nose mask.
VIEWS = {
    "frontal_0": ("frontal_0.png", frontal_broad_nose),
    "left20": ("left20.png", profile_broad_nose),
    "left40": ("left40.png", profile_broad_nose),
    "left60": ("left60.png", profile_broad_nose),
    "right20": ("right20.png", profile_broad_nose),
    "right40": ("right40.png", profile_broad_nose),
    "right60": ("right60.png", profile_broad_nose),
}


def main():
    for view_name, (image_name, deform_function) in VIEWS.items():
        image_path = INPUT_DIR / image_name
        mask_path = MASK_DIR / view_name / "nose_binary.png"

        print(f"\nProcessing: {view_name}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"SKIPPED: Cannot read image: {image_path}")
            continue

        if not mask_path.exists():
            print(f"SKIPPED: Mask not found: {mask_path}")
            continue

        nose_mask = load_nose_mask(
            str(mask_path),
            image.shape[:2]
        )

        result = deform_function(
            image,
            nose_mask,
            strength=0.25
        )

        view_output_dir = OUTPUT_DIR / view_name
        view_output_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(view_output_dir / "original.png"), image)
        cv2.imwrite(str(view_output_dir / "nose_mask.png"), nose_mask)
        cv2.imwrite(str(view_output_dir / "broad_nose.png"), result)

        print(f"Saved results to: {view_output_dir}")

    print("\nAll available frontal and angled-view tests completed.")


if __name__ == "__main__":
    main()
