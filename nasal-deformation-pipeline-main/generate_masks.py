
from pathlib import Path
import argparse
import sys

import cv2
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
BISENET_DIR = ROOT / "models" / "bisenet"
IDENTITIES_DIR = ROOT / "data" / "identities"

WEIGHT_PATH = BISENET_DIR / "weights" / "resnet18.pt"
NOSE_CLASS_ID = 10
INPUT_SIZE = (512, 512)

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

# Conservative expansion is enabled ONLY for these views.
EXPANSION_VIEW_NAMES = {"left60", "right60"}

# Small elliptical kernel: deliberately conservative.
EXPANSION_KERNEL_SIZE = 3
SMOOTH_KERNEL_SIZE = 3

sys.path.insert(0, str(BISENET_DIR))
from inference import load_model, prepare_image  # noqa: E402


# ---------------------------------------------------------
# View-name detection
# ---------------------------------------------------------
def identify_view(stem):
    """Recognize both descriptive and numbered view filenames."""
    value = "".join(ch.lower() for ch in stem if ch.isalnum())

    while value and value[0].isdigit():
        value = value[1:]

    aliases = {
        "front": "frontal",
        "frontal": "frontal",
        "frontal0": "frontal",
        "left20": "left20",
        "left40": "left40",
        "left60": "left60",
        "right20": "right20",
        "right40": "right40",
        "right60": "right60",
        "basal": "basal",
        "basalview": "basal",
        "helicopter": "helicopter",
        "helicopterview": "helicopter",
    }

    return aliases.get(value)


# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------
def make_overlay(image_bgr, nose_mask):
    """Highlight mask pixels in red."""
    overlay = image_bgr.copy()
    selected = nose_mask > 0

    red = np.zeros_like(image_bgr)
    red[:, :] = (0, 0, 255)

    overlay[selected] = (
        0.45 * image_bgr[selected].astype(np.float32)
        + 0.55 * red[selected].astype(np.float32)
    ).astype(np.uint8)

    return overlay


def make_label_visualization(labels):
    """Colorize the original BiSeNet class-ID map."""
    visualization = np.zeros(
        (labels.shape[0], labels.shape[1], 3),
        dtype=np.uint8,
    )

    for class_id in range(19):
        color = (
            (class_id * 37 + 50) % 256,
            (class_id * 67 + 90) % 256,
            (class_id * 97 + 130) % 256,
        )
        visualization[labels == class_id] = color

    return visualization


# ---------------------------------------------------------
# Conservative mask processing
# ---------------------------------------------------------
def keep_relevant_components(binary):
    """
    Preserve the main component and nearby small components.

    Do not fill holes. Do not retain distant isolated predictions,
    which may be segmentation noise.
    """
    binary = (binary > 0).astype(np.uint8)

    count, component_labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    if count <= 1:
        return binary * 255

    areas = stats[1:, cv2.CC_STAT_AREA]

    if areas.size == 0:
        return binary * 255

    main_label = 1 + int(np.argmax(areas))
    main_area = int(stats[main_label, cv2.CC_STAT_AREA])

    main_component = (
        component_labels == main_label
    ).astype(np.uint8)

    main_centroid = centroids[main_label]
    main_x, main_y = main_centroid

    # Preserve only small components close to the main component.
    # This avoids retaining unrelated isolated nose-class pixels.
    h, w = binary.shape
    max_distance = max(5.0, 0.035 * min(h, w))
    min_component_area = max(2, int(main_area * 0.01))

    cleaned = main_component.copy()

    for label in range(1, count):
        if label == main_label:
            continue

        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < min_component_area:
            continue

        cx, cy = centroids[label]
        distance = float(
            np.hypot(cx - main_x, cy - main_y)
        )

        if distance <= max_distance:
            cleaned[component_labels == label] = 1

    return cleaned * 255


def clean_nose_mask(raw_nose_mask, view_name):
    """
    Clean a nose mask without landmark detection.

    All views:
      - Preserve the raw prediction separately.
      - Retain the main component and nearby meaningful components.
      - Apply only minimal smoothing.

    Left60 and right60 only:
      - Allow a single small dilation to reconnect a narrow missing
        boundary region.
      - Expansion is restricted to these two views.
    """
    if (
        not isinstance(raw_nose_mask, np.ndarray)
        or raw_nose_mask.ndim != 2
    ):
        raise ValueError("raw_nose_mask must be a 2D NumPy array.")

    binary = (raw_nose_mask > 0).astype(np.uint8) * 255

    if not np.any(binary):
        return binary

    cleaned = keep_relevant_components(binary)

    # Minimal boundary smoothing, without unconditional hole filling.
    smooth_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (SMOOTH_KERNEL_SIZE, SMOOTH_KERNEL_SIZE),
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        smooth_kernel,
        iterations=1,
    )

    # Expansion is restricted to the two 60-degree views.
    if view_name in EXPANSION_VIEW_NAMES:
        expansion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (EXPANSION_KERNEL_SIZE, EXPANSION_KERNEL_SIZE),
        )

        cleaned = cv2.dilate(
            cleaned,
            expansion_kernel,
            iterations=1,
        )

    # Restore a clean binary mask.
    return np.where(cleaned > 0, 255, 0).astype(np.uint8)


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate BiSeNet nose masks for one identity."
    )
    parser.add_argument(
        "--identity",
        required=True,
        help="Identity folder, e.g. H01, H02, or H09.",
    )
    return parser.parse_args()


# ---------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------
def main():
    args = parse_args()

    identity_dir = IDENTITIES_DIR / args.identity
    input_dir = identity_dir / "input"
    output_dir = identity_dir / "masks"

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(
            f"BiSeNet weights not found: {WEIGHT_PATH}"
        )

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Identity: {args.identity}")
    print(f"Device: {device}")
    print(f"Input images: {len(image_paths)}")
    print("Landmark detection: disabled")
    print("Mask expansion: left60 and right60 only")

    model = load_model(
        model_name="resnet18",
        num_classes=19,
        weight_path=str(WEIGHT_PATH),
        device=device,
    )

    model.eval()

    successes = 0
    failures = []

    for image_path in image_paths:
        try:
            view_name = identify_view(image_path.stem)

            with Image.open(image_path) as source:
                image_rgb = source.convert("RGB")
                original_size = image_rgb.size

                tensor = prepare_image(
                    image_rgb,
                    input_size=INPUT_SIZE,
                ).to(device)

            with torch.no_grad():
                model_output = model(tensor)[0]

                labels_512 = (
                    model_output.squeeze(0)
                    .argmax(dim=0)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

            # Restore class IDs to original image dimensions.
            label_image = Image.fromarray(labels_512)
            label_image = label_image.resize(
                original_size,
                resample=Image.Resampling.NEAREST,
            )

            labels = np.asarray(
                label_image,
                dtype=np.uint8,
            )

            raw_nose_mask = (
                labels == NOSE_CLASS_ID
            ).astype(np.uint8) * 255

            nose_mask = clean_nose_mask(
                raw_nose_mask,
                view_name,
            )

            original_bgr = cv2.cvtColor(
                np.asarray(image_rgb),
                cv2.COLOR_RGB2BGR,
            )

            view_dir = output_dir / image_path.stem
            view_dir.mkdir(parents=True, exist_ok=True)

            # Save the original class-ID prediction.
            labels_path = view_dir / "labels.png"

            # Save a colored visualization of all classes.
            visualization_path = (
                view_dir / "labels_visualization.png"
            )

            # Save both raw and corrected binary masks.
            raw_mask_path = view_dir / "nose_raw.png"
            binary_path = view_dir / "nose_binary.png"

            # Save overlays for visual inspection.
            raw_overlay_path = view_dir / "nose_raw_overlay.png"
            overlay_path = view_dir / "nose_overlay.png"

            files_to_save = [
                (labels_path, labels),
                (
                    visualization_path,
                    make_label_visualization(labels),
                ),
                (raw_mask_path, raw_nose_mask),
                (binary_path, nose_mask),
                (
                    raw_overlay_path,
                    make_overlay(original_bgr, raw_nose_mask),
                ),
                (
                    overlay_path,
                    make_overlay(original_bgr, nose_mask),
                ),
            ]

            for path, array in files_to_save:
                if not cv2.imwrite(str(path), array):
                    raise OSError(f"Could not save {path}")

            raw_pixels = int(
                np.count_nonzero(raw_nose_mask)
            )
            corrected_pixels = int(
                np.count_nonzero(nose_mask)
            )

            print(
                f"[OK] {image_path.name}: "
                f"view={view_name or 'unrecognized'}, "
                f"raw_pixels={raw_pixels}, "
                f"corrected_pixels={corrected_pixels}, "
                f"expansion={'yes' if view_name in EXPANSION_VIEW_NAMES else 'no'}"
            )

            if corrected_pixels == 0:
                print(
                    "  WARNING: empty mask. Inspect labels.png "
                    "and nose_raw.png."
                )

            successes += 1

        except Exception as exc:
            failures.append((image_path.name, str(exc)))
            print(f"[FAILED] {image_path.name}: {exc}")

    print("\nMask generation finished.")
    print(f"Successful: {successes}/{len(image_paths)}")
    print(f"Output directory: {output_dir}")

    if failures:
        print("\nFailed images:")
        for name, error in failures:
            print(f"  {name}: {error}")


if __name__ == "__main__":
    main()
