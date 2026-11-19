
import cv2
import numpy as np


NOSE_CLASS_ID = 10


def prepare_nose_mask(mask, image_shape, nose_class_id=NOSE_CLASS_ID):
    """
    Convert a BiSeNet indexed-label map or binary mask into a
    binary uint8 nose mask matching the input image dimensions.

    Parameters
    ----------
    mask : np.ndarray
        BiSeNet indexed labels or an existing binary nose mask.
    image_shape : tuple
        Image shape, either (height, width) or (height, width, channels).
    nose_class_id : int
        Nose class ID in the BiSeNet indexed-label map.

    Returns
    -------
    np.ndarray
        Binary uint8 mask: 255 for nose pixels, 0 elsewhere.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a NumPy array.")

    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("mask must be a nonempty 2D array.")

    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width.")

    height, width = int(image_shape[0]), int(image_shape[1])

    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")

    # Preserve the convention used by this project's BiSeNet:
    # indexed nose label = 10.
    unique_values = np.unique(mask)

    if np.all(np.isin(unique_values, [0, 1, 255])):
        binary = mask > 0
    else:
        binary = mask == nose_class_id

    binary = binary.astype(np.uint8)

    if binary.shape != (height, width):
        binary = cv2.resize(
            binary,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    if not np.any(binary):
        raise ValueError(
            "The nose mask is empty. Check whether the input is a "
            "binary mask or an indexed BiSeNet label map."
        )

    return binary * 255


def load_nose_mask(mask_path, image_shape, nose_class_id=NOSE_CLASS_ID):
    """
    Load a saved mask image and prepare it for deformation.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return prepare_nose_mask(mask, image_shape, nose_class_id)


def save_nose_mask(mask, output_path):
    """
    Save a binary nose mask as a grayscale PNG.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a 2D NumPy array.")

    binary = (mask > 0).astype(np.uint8) * 255

    if not cv2.imwrite(output_path, binary):
        raise OSError(f"Could not save nose mask: {output_path}")
