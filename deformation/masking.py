"""
Nasal Masking & Segmentation Utilities.

Provides functions to load, prepare, clean, and validate binary nasal masks
for 2D morphology deformation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

NOSE_CLASS_ID = 10
EXPANSION_VIEW_NAMES = {"left60", "right60", "left_60", "right_60"}
EXPANSION_KERNEL_SIZE = 3
SMOOTH_KERNEL_SIZE = 3


def keep_relevant_components(binary: np.ndarray) -> np.ndarray:
    """
    Preserve the main component and nearby small components.
    Removes distant isolated segmentation noise.
    """
    binary = (binary > 0).astype(np.uint8)

    count, component_labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if count <= 1:
        return binary * 255

    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return binary * 255

    main_label = 1 + int(np.argmax(areas))
    main_area = int(stats[main_label, cv2.CC_STAT_AREA])

    main_component = (component_labels == main_label).astype(np.uint8)
    main_centroid = centroids[main_label]
    main_x, main_y = main_centroid

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
        distance = float(np.hypot(cx - main_x, cy - main_y))

        if distance <= max_distance:
            cleaned[component_labels == label] = 1

    return cleaned * 255


def clean_nose_mask(raw_nose_mask: np.ndarray, view_name: str = "frontal") -> np.ndarray:
    """
    Clean a raw nose mask:
      - Preserve the main component and nearby meaningful components.
      - Apply minimal elliptical closing.
      - For left60/right60 views, apply small conservative dilation.
    """
    if not isinstance(raw_nose_mask, np.ndarray) or raw_nose_mask.ndim != 2:
        raise ValueError("raw_nose_mask must be a 2D NumPy array.")

    binary = (raw_nose_mask > 0).astype(np.uint8) * 255

    if not np.any(binary):
        return binary

    cleaned = keep_relevant_components(binary)

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

    norm_view = view_name.lower().replace("_", "")
    if norm_view in {"left60", "right60"}:
        expansion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (EXPANSION_KERNEL_SIZE, EXPANSION_KERNEL_SIZE),
        )
        cleaned = cv2.dilate(
            cleaned,
            expansion_kernel,
            iterations=1,
        )

    return np.where(cleaned > 0, 255, 0).astype(np.uint8)


def prepare_nose_mask(
    mask: np.ndarray,
    image_shape: Tuple[int, ...],
    nose_class_id: int = NOSE_CLASS_ID,
) -> np.ndarray:
    """
    Convert an indexed BiSeNet label map or binary mask into a
    binary uint8 nose mask matching image_shape (height, width).
    Returns 255 for nose pixels, 0 elsewhere.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a NumPy array.")

    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("mask must be a nonempty 2D array.")

    if len(image_shape) < 2:
        raise ValueError("image_shape must contain at least (height, width).")

    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")

    unique_values = np.unique(mask)

    if np.all(np.isin(unique_values, [0, 1, 255])):
        binary = (mask > 0).astype(np.uint8)
    else:
        binary = (mask == nose_class_id).astype(np.uint8)

    if binary.shape != (height, width):
        binary = cv2.resize(
            binary,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    if not np.any(binary):
        raise ValueError(
            "The nose mask is empty. Check whether input is valid."
        )

    return binary * 255


def load_nose_mask(
    mask_path: Union[str, Path],
    image_shape: Tuple[int, ...],
    nose_class_id: int = NOSE_CLASS_ID,
) -> np.ndarray:
    """
    Load a saved mask file and prepare it for deformation.
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to decode mask: {mask_path}")

    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return prepare_nose_mask(mask, image_shape, nose_class_id)


def save_nose_mask(mask: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Save a binary nose mask as a grayscale PNG (255=nose, 0=background).
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a 2D NumPy array.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    binary = (mask > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(output_path), binary):
        raise OSError(f"Could not save nose mask to: {output_path}")


def make_overlay(image_bgr: np.ndarray, nose_mask: np.ndarray) -> np.ndarray:
    """
    Generate an inspection overlay highlighting nose mask pixels in translucent red.
    """
    overlay = image_bgr.copy()
    selected = nose_mask > 0

    red = np.zeros_like(image_bgr)
    red[:, :] = (0, 0, 255)

    overlay[selected] = (
        0.45 * image_bgr[selected].astype(np.float32)
        + 0.55 * red[selected].astype(np.float32)
    ).astype(np.uint8)

    return overlay

