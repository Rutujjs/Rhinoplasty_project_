
"""
Main controller for nasal-deformation-pipeline.

Run from the project root:
    python main.py --identity H06 --deformation saddle
    python main.py --identity H02 --deformation broad
    python main.py --identity H06 --deformation deviated --direction left
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import re

ROOT = Path(__file__).resolve().parent
IDENTITIES_DIR = ROOT / "data" / "identities"
OUTPUT_DIR = ROOT / "data" / "output"
MASK_GENERATOR = ROOT / "generate_masks.py"

VIEWS = [
    "frontal",
    "left20",
    "left40",
    "left60",
    "right20",
    "right40",
    "right60",
    "basal",
    "helicopter",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalize_stem(stem: str) -> str:
    """Convert a filename stem to lowercase alphanumeric text."""
    return "".join(ch.lower() for ch in stem if ch.isalnum())


def identify_view(stem: str) -> str | None:
    """
    Identify a view from common filename styles, including:
    01_frontal_0.png
    Front.png
    H09_Left_40_View.png
    Left40.png
    right_60_degree.png
    Basal_View.png
    Helicopter.png
    """
    value = normalize_stem(stem)

    # Remove common identity prefixes, such as H09.
    value = re.sub(r"^h\d+", "", value)

    # Remove a leading numeric image index, e.g. 01frontal0.
    value = re.sub(r"^\d+", "", value)

    # Match distinctive view names first.
    if "helicopter" in value:
        return "helicopter"

    if "basal" in value:
        return "basal"

    # Match angle-specific views.
    angle_patterns = [
        (r"left(?:view)?20(?:degree|degrees|deg)?", "left20"),
        (r"left(?:view)?40(?:degree|degrees|deg)?", "left40"),
        (r"left(?:view)?60(?:degree|degrees|deg)?", "left60"),
        (r"right(?:view)?20(?:degree|degrees|deg)?", "right20"),
        (r"right(?:view)?40(?:degree|degrees|deg)?", "right40"),
        (r"right(?:view)?60(?:degree|degrees|deg)?", "right60"),
    ]

    for pattern, view_name in angle_patterns:
        if re.search(pattern, value):
            return view_name

    # Match frontal views without accidentally matching an angled view.
    if "frontal" in value or "front" in value:
        return "frontal"

    return None


def discover_input_images(input_dir: Path) -> dict[str, Path]:
    """Find one image per view."""
    found = {}

    if not input_dir.is_dir():
        return found

    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        view = identify_view(path.stem)

        if view:
            if view in found:
                raise ValueError(
                    f"Multiple images match view '{view}': "
                    f"{found[view].name} and {path.name}"
                )

            found[view] = path

    return found


def load_deformation_functions():
    """Import the existing deformation functions without modifying them."""
    try:
        from src.deformation import (
            basal_broad_nose,
            basal_deviated_nose_deformation,
            basal_hump_nose,
            basal_saddle_nose,
            broad_nose_deformation,
            deviated_nose_deformation,
            frontal_broad_nose,
            frontal_hump_nose,
            frontal_saddle_nose,
            helicopter_broad_nose,
            helicopter_deviated_nose_deformation,
            helicopter_hump_nose,
            helicopter_saddle_nose,
            hump_nose_deformation,
            profile_broad_nose,
            saddle_nose_deformation,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Could not import functions from src/deformation.py. "
            "Run this script from the project root and check your module structure."
        ) from exc

    return {
        "saddle": {
            "frontal": lambda i, m, d: frontal_saddle_nose(
                i, m, strength=0.65
            ),
            "left20": lambda i, m, d: saddle_nose_deformation(
                i, m, strength=0.65, contour_side="left"
            ),
            "left40": lambda i, m, d: saddle_nose_deformation(
                i, m, strength=0.65, contour_side="left"
            ),
            "left60": lambda i, m, d: saddle_nose_deformation(
                i, m, strength=0.65, contour_side="left"
            ),
            "right20": lambda i, m, d: saddle_nose_deformation(
                i, m, strength=0.65, contour_side="right"
            ),
            "right40": lambda i, m, d: saddle_nose_deformation(
                i, m, strength=0.65, contour_side="right"
            ),
            "right60": lambda i, m, d: saddle_nose_deformation(
                i, m, strength=0.65, contour_side="right"
            ),
            "basal": lambda i, m, d: basal_saddle_nose(
                i, m, strength=0.65
            ),
            "helicopter": lambda i, m, d: helicopter_saddle_nose(
                i, m, strength=0.65
            ),
        },

        "hump": {
            "frontal": lambda i, m, d: frontal_hump_nose(
                i, m, strength=0.65
            ),
            "left20": lambda i, m, d: hump_nose_deformation(
                i, m, strength=0.40, contour_side="left"
            ),
            "left40": lambda i, m, d: hump_nose_deformation(
                i, m, strength=0.40, contour_side="left"
            ),
            "left60": lambda i, m, d: hump_nose_deformation(
                i, m, strength=0.40, contour_side="left"
            ),
            "right20": lambda i, m, d: hump_nose_deformation(
                i, m, strength=0.40, contour_side="right"
            ),
            "right40": lambda i, m, d: hump_nose_deformation(
                i, m, strength=0.40, contour_side="right"
            ),
            "right60": lambda i, m, d: hump_nose_deformation(
                i, m, strength=0.40, contour_side="right"
            ),
            "basal": lambda i, m, d: basal_hump_nose(
                i, m, strength=0.25
            ),
            "helicopter": lambda i, m, d: helicopter_hump_nose(
                i, m, strength=0.25
            ),
        },

        "deviated": {
           "frontal": lambda i, m, d: deviated_nose_deformation(
                i, m, strength=0.40, direction=d
            ),
            "left20": lambda i, m, d: deviated_nose_deformation(
                i, m, strength=0.30, direction=d
            ),
            "left40": lambda i, m, d: deviated_nose_deformation(
                i, m, strength=0.30, direction=d
            ),
            "left60": lambda i, m, d: deviated_nose_deformation(
                i, m, strength=0.30, direction=d
            ),
            "right20": lambda i, m, d: deviated_nose_deformation(
                i, m, strength=0.30, direction=d
            ),
            "right40": lambda i, m, d: deviated_nose_deformation(
                i, m, strength=0.30, direction=d
            ),
            "right60": lambda i, m, d: deviated_nose_deformation(
                i, m, strength=0.30, direction=d
            ),
            "basal": lambda i, m, d: basal_deviated_nose_deformation(
                i, m, strength=0.25, direction=d
            ),
            "helicopter": lambda i, m, d: helicopter_deviated_nose_deformation(
                i, m, strength=0.35, direction=d
            ),
        },

        "broad": {
            "frontal": lambda i, m, d: frontal_broad_nose(
                i, m, strength=0.40
            ),
            "left20": lambda i, m, d: profile_broad_nose(
                i, m, strength=0.40
            ),
            "left40": lambda i, m, d: profile_broad_nose(
                i, m, strength=0.40
            ),
            "left60": lambda i, m, d: profile_broad_nose(
                i, m, strength=0.40
            ),
            "right20": lambda i, m, d: profile_broad_nose(
                i, m, strength=0.40
            ),
            "right40": lambda i, m, d: profile_broad_nose(
                i, m, strength=0.40
            ),
            "right60": lambda i, m, d: profile_broad_nose(
                i, m, strength=0.40
            ),
            "basal": lambda i, m, d: basal_broad_nose(
                i, m, strength=0.40
            ),
            "helicopter": lambda i, m, d: helicopter_broad_nose(
                i, m, strength=0.40
            ),
        },
    }


def ensure_masks(identity, input_dir, masks_dir, image_by_view):
    """Generate masks if the expected nose_binary.png files are missing."""
    missing_views = [
        view for view in VIEWS if view not in image_by_view
    ]

    if missing_views:
        raise FileNotFoundError(
            "Could not identify input images for: "
            + ", ".join(missing_views)
            + f"\nInput folder: {input_dir}"
        )

    missing = [
        view for view in VIEWS
        if not (
            masks_dir
            / image_by_view[view].stem
            / "nose_binary.png"
        ).is_file()
    ]

    if not missing:
        return

    if not MASK_GENERATOR.is_file():
        raise FileNotFoundError(
            f"Mask generator not found: {MASK_GENERATOR}"
        )

    print(
        f"\nNose masks missing for {identity}; "
        "running generate_masks.py..."
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(MASK_GENERATOR),
            "--identity",
            identity,
        ],
        cwd=str(ROOT),
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "generate_masks.py failed. Check the error printed above."
        )

    still_missing = [
        view for view in VIEWS
        if not (
            masks_dir
            / image_by_view[view].stem
            / "nose_binary.png"
        ).is_file()
    ]

    if still_missing:
        raise FileNotFoundError(
            "Masks still missing for views: "
            + ", ".join(still_missing)
        )


def process_identity(identity, deformation, direction="right"):
    identity_dir = IDENTITIES_DIR / identity
    input_dir = identity_dir / "input"
    masks_dir = identity_dir / "masks"
    output_dir = OUTPUT_DIR / identity / deformation

    if not identity_dir.is_dir():
        available = (
            sorted(p.name for p in IDENTITIES_DIR.iterdir() if p.is_dir())
            if IDENTITIES_DIR.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"Identity folder not found: {identity_dir}\n"
            f"Available identities: {', '.join(available) or '(none)'}"
        )

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    image_by_view = discover_input_images(input_dir)

    ensure_masks(
        identity, input_dir, masks_dir, image_by_view
    )

    functions = load_deformation_functions()[deformation]
    output_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failures = []

    print(f"\nIdentity: {identity}")
    print(f"Deformation: {deformation}")
    print(f"Output folder: {output_dir}\n")

    for view in VIEWS:
        image_path = image_by_view[view]
        mask_path = (
            masks_dir / image_path.stem / "nose_binary.png"
        )

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            failures.append(f"{view}: cannot read {image_path}")
            print(f"[READ ERROR] {image_path}")
            continue

        if mask is None:
            failures.append(f"{view}: cannot read {mask_path}")
            print(f"[READ ERROR] {mask_path}")
            continue

        mask = np.where(mask > 0, 255, 0).astype(np.uint8)

        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        try:
            result = functions[view](image, mask, direction)

            if not isinstance(result, np.ndarray):
                raise ValueError(
                    "Deformation function did not return a NumPy image."
                )

            if result.shape != image.shape:
                raise ValueError(
                    f"Unexpected output shape: {result.shape}; "
                    f"expected {image.shape}"
                )

            if result.dtype != np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)

            destination = output_dir / image_path.name

            if not cv2.imwrite(str(destination), result):
                raise OSError(f"Could not write {destination}")

            succeeded += 1
            print(f"[OK] {view:10s} -> {destination}")

        except Exception as exc:
            failures.append(
                f"{view}: {type(exc).__name__}: {exc}"
            )
            print(
                f"[FAILED] {view}: {type(exc).__name__}: {exc}"
            )

    print(f"\nFinished: {succeeded}/9 views saved.")

    if failures:
        print("\nIssues:")
        for failure in failures:
            print(f" - {failure}")

    print(f"\nResults: {output_dir}")

    return 0 if succeeded == 9 else 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a nasal deformation to all nine views."
    )

    parser.add_argument(
        "--identity",
        required=True,
        help="Identity folder, e.g. H01 or H06",
    )

    parser.add_argument(
        "--deformation",
        required=True,
        choices=["saddle", "hump", "deviated", "broad"],
    )

    parser.add_argument(
        "--direction",
        choices=["left", "right"],
        default="right",
        help="Deviation direction; only used for deviated deformation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        return process_identity(
            args.identity,
            args.deformation,
            args.direction,
        )
    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
