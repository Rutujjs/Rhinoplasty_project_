
import cv2
import numpy as np



def saddle_nose_deformation(
    image,
    nose_mask,
    strength=0.65,
    contour_side="left",
):
    """Smooth, localized saddle-nose deformation for angled views."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a nonempty BGR image (H, W, 3).")

    if not isinstance(nose_mask, np.ndarray) or nose_mask.ndim != 2:
        raise ValueError("nose_mask must be a 2D NumPy array.")

    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")

    if contour_side not in ("left", "right"):
        raise ValueError("contour_side must be 'left' or 'right'.")

    h, w = image.shape[:2]

    if nose_mask.shape != (h, w):
        nose_mask = cv2.resize(
            nose_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = (nose_mask > 0).astype(np.uint8)

    if not np.any(mask):
        raise ValueError("The nose mask is empty.")

    ys, xs = np.where(mask > 0)
    y_top, y_bottom = int(ys.min()), int(ys.max())
    x_left, x_right = int(xs.min()), int(xs.max())

    nose_width = x_right - x_left + 1
    nose_height = y_bottom - y_top + 1

    if nose_width < 3 or nose_height < 8:
        raise ValueError("The nose mask is too small.")

    # Estimate the selected visible contour on each row.
    contour = np.full(h, np.nan, dtype=np.float32)

    for y in range(y_top, y_bottom + 1):
        row_x = np.flatnonzero(mask[y])
        if row_x.size:
            contour[y] = (
                float(row_x.min())
                if contour_side == "left"
                else float(row_x.max())
            )

    valid_rows = np.flatnonzero(np.isfinite(contour))

    if valid_rows.size < 5:
        raise ValueError("Not enough valid nose contour rows.")

    contour = np.interp(
        np.arange(h),
        valid_rows,
        contour[valid_rows],
    ).astype(np.float32)

    contour = cv2.GaussianBlur(
        contour.reshape(-1, 1),
        (0, 0),
        sigmaX=max(1.0, nose_height * 0.025),
    ).ravel()

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Smooth vertical profile: minimal movement at the root and tip.
    t = np.clip(
        (np.arange(h, dtype=np.float32) - y_top)
        / max(nose_height - 1, 1),
        0.0,
        1.0,
    )

    bridge_profile = np.exp(
        -0.5 * ((t - 0.43) / 0.20) ** 2
    )

    endpoint_taper = np.sin(np.pi * t).clip(0.0, 1.0) ** 1.5
    profile = (bridge_profile * endpoint_taper).astype(np.float32)

    signed_distance = xx - contour[:, None]

    if contour_side == "left":
        face_distance = signed_distance
        direction = 1.0
    else:
        face_distance = -signed_distance
        direction = -1.0

    support_width = max(3.0, nose_width * 0.30)

    contour_band = np.exp(
        -0.5 * (face_distance / support_width) ** 2
    ).astype(np.float32)

    # Keep the displacement conservative and smooth.
    max_shift = strength * nose_width * 0.25

    displacement = (
        direction
        * max_shift
        * profile[:, None]
        * contour_band
    ).astype(np.float32)

    # Smooth the displacement itself to prevent local kinks.
    displacement = cv2.GaussianBlur(
        displacement,
        (0, 0),
        sigmaX=max(1.0, nose_width * 0.06),
        sigmaY=max(1.0, nose_height * 0.015),
    )

    # Feather only inward from the original mask boundary.
    distance_inside = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5,
    )

    feather_width = max(2.0, nose_width * 0.12)

    alpha = np.clip(
        distance_inside / feather_width,
        0.0,
        1.0,
    ).astype(np.float32)

    alpha = cv2.GaussianBlur(
        alpha,
        (0, 0),
        sigmaX=max(0.8, nose_width * 0.015),
    )

    # Never allow the blending mask to affect pixels outside the nose.
    alpha *= mask.astype(np.float32)

    displacement *= alpha

    # Backward remapping: each output pixel samples a source location.
    map_x = xx - displacement
    map_y = yy.copy()

    # Do not reject boundary pixels abruptly. The smooth alpha taper
    # reduces movement near the boundary and avoids abrupt reversions.
    warped = cv2.remap(
        image,
        np.ascontiguousarray(map_x, dtype=np.float32),
        np.ascontiguousarray(map_y, dtype=np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )

    # Use the feathered alpha only to control displacement.
# Do not blend the original and warped images.
    result = image.copy()

    inside = mask > 0

    result[inside] = warped[inside]

    # Preserve every pixel outside the original nose mask.
    result[~inside] = image[~inside]

    return result


# Keep compatibility with existing pipeline calls.
def left60_saddle_nose(image, nose_mask, strength=0.65):
    return saddle_nose_deformation(
        image,
        nose_mask,
        strength=strength,
        contour_side="left",
    )



def frontal_saddle_nose(image, nose_mask, strength=0.65):
    """
    Apply a symmetric, localized saddle-like depression to the
    frontal nasal bridge using the existing binary nose mask.

    This is a 2D approximation. Visual validation is required.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must have shape (H, W, 3).")

    if not isinstance(nose_mask, np.ndarray) or nose_mask.ndim != 2:
        raise ValueError("nose_mask must be a 2D NumPy array.")

    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")

    h, w = image.shape[:2]

    if nose_mask.shape != (h, w):
        nose_mask = cv2.resize(
            nose_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = (nose_mask > 0).astype(np.uint8)

    if not np.any(mask):
        raise ValueError("The nose mask is empty.")

    ys, xs = np.where(mask > 0)

    y_top, y_bottom = int(ys.min()), int(ys.max())
    x_left, x_right = int(xs.min()), int(xs.max())

    nose_height = y_bottom - y_top + 1
    nose_width = x_right - x_left + 1

    if nose_height < 8 or nose_width < 3:
        raise ValueError("The nose mask is too small.")

    # Estimate the center of the nose on each row.
    # This follows the mask rather than assuming a perfectly
    # vertical centerline.
    centerline = np.full(h, np.nan, dtype=np.float32)

    for y in range(y_top, y_bottom + 1):
        row_x = np.flatnonzero(mask[y])

        if row_x.size:
            centerline[y] = (row_x.min() + row_x.max()) / 2.0

    valid_rows = np.flatnonzero(np.isfinite(centerline))

    if valid_rows.size < 5:
        raise ValueError("Unable to estimate the nasal centerline.")

    centerline = np.interp(
        np.arange(h),
        valid_rows,
        centerline[valid_rows],
    ).astype(np.float32)

    centerline = cv2.GaussianBlur(
        centerline.reshape(-1, 1),
        (0, 0),
        sigmaX=max(1.0, nose_height * 0.015),
    ).ravel()

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Normalized vertical position within the nose.
    t = np.clip(
        (np.arange(h, dtype=np.float32) - y_top)
        / max(nose_height - 1, 1),
        0.0,
        1.0,
    )

    # Localize the effect to the upper-middle bridge.
    # Taper the displacement near the root and tip.
    bridge_profile = np.exp(
        -0.5 * ((t - 0.43) / 0.19) ** 2
    )

    endpoint_taper = np.clip(
        np.sin(np.pi * t), 0.0, 1.0
    ) ** 1.2

    profile = (bridge_profile * endpoint_taper).astype(np.float32)

    # Symmetric horizontal influence around the estimated centerline.
    horizontal_distance = xx - centerline[:, None]
    sigma = max(2.0, nose_width * 0.35)

    horizontal_profile = np.exp(
        -0.5 * (horizontal_distance / sigma) ** 2
    ).astype(np.float32)

    # A conservative localized warp. The sign and magnitude are
    # starting parameters, not a calibrated anatomical measurement.
    max_shift = strength * nose_width * 0.12

    displacement = (
        max_shift
        * profile[:, None]
        * horizontal_profile
    ).astype(np.float32)

    # Feather displacement inward from the original nose boundary.
    distance_inside = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5,
    )

    feather_width = max(2.0, nose_width * 0.12)

    alpha = np.clip(
        (distance_inside - 1.0) / feather_width,
        0.0,
        1.0,
    ).astype(np.float32)

    alpha = cv2.GaussianBlur(
        alpha,
        (0, 0),
        sigmaX=max(0.8, nose_width * 0.015),
    )

    alpha *= mask.astype(np.float32)

    displacement *= alpha

    map_x = np.ascontiguousarray(
        xx - np.sign(horizontal_distance) * displacement,
        dtype=np.float32,
    )

    map_y = np.ascontiguousarray(yy, dtype=np.float32)

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )

    result = image.copy()
    inside = mask > 0
    result[inside] = warped[inside]

    return result


def helicopter_saddle_nose(image, nose_mask, strength=0.65):
    """Apply a localized, symmetric bridge-shape approximation from above."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must have shape (H, W, 3).")

    if not isinstance(nose_mask, np.ndarray) or nose_mask.ndim != 2:
        raise ValueError("nose_mask must be a 2D NumPy array.")

    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")

    h, w = image.shape[:2]

    if nose_mask.shape != (h, w):
        nose_mask = cv2.resize(
            nose_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = (nose_mask > 0).astype(np.uint8)

    if not np.any(mask):
        raise ValueError("The nose mask is empty.")

    ys, xs = np.where(mask > 0)
    y_top, y_bottom = int(ys.min()), int(ys.max())
    x_left, x_right = int(xs.min()), int(xs.max())

    nose_height = y_bottom - y_top + 1
    nose_width = x_right - x_left + 1

    if nose_height < 8 or nose_width < 3:
        raise ValueError("The nose mask is too small.")

    # Estimate a centerline for each row.
    centerline = np.full(h, np.nan, dtype=np.float32)

    for y in range(y_top, y_bottom + 1):
        row_x = np.flatnonzero(mask[y])
        if row_x.size:
            centerline[y] = (row_x.min() + row_x.max()) / 2.0

    valid = np.flatnonzero(np.isfinite(centerline))

    if valid.size < 5:
        raise ValueError("Unable to estimate the nose centerline.")

    centerline = np.interp(
        np.arange(h), valid, centerline[valid]
    ).astype(np.float32)

    centerline = cv2.GaussianBlur(
        centerline.reshape(-1, 1),
        (0, 0),
        sigmaX=max(1.0, nose_height * 0.015),
    ).ravel()

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    t = np.clip(
        (np.arange(h, dtype=np.float32) - y_top)
        / max(nose_height - 1, 1),
        0.0,
        1.0,
    )

    # Restrict the effect to the upper-middle portion of the nose.
    bridge_profile = np.exp(
        -0.5 * ((t - 0.43) / 0.20) ** 2
    )

    taper = np.clip(np.sin(np.pi * t), 0.0, 1.0) ** 1.2
    profile = (bridge_profile * taper).astype(np.float32)

    horizontal_distance = xx - centerline[:, None]

    # A smooth inward mapping produces a subtle narrowing.
    amount = 0.12 * strength * profile[:, None]

    map_x = (
        centerline[:, None]
        + horizontal_distance * (1.0 + amount)
    ).astype(np.float32)

    map_y = yy.copy()

    # Feather displacement inward from the original mask.
    distance_inside = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5,
    )

    feather_width = max(2.0, nose_width * 0.12)

    alpha = np.clip(
        (distance_inside - 1.0) / feather_width,
        0.0,
        1.0,
    ).astype(np.float32)

    alpha = cv2.GaussianBlur(
        alpha,
        (0, 0),
        sigmaX=max(0.8, nose_width * 0.015),
    )

    alpha *= mask.astype(np.float32)

    map_x = xx + (map_x - xx) * alpha

    warped = cv2.remap(
        image,
        np.ascontiguousarray(map_x, dtype=np.float32),
        np.ascontiguousarray(map_y, dtype=np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )

    result = image.copy()
    inside = mask > 0
    result[inside] = warped[inside]

    return result


def basal_saddle_nose(image, nose_mask, strength=0.65):
    """
    Preserve the basal view for saddle-nose datasets.

    The underside view does not reliably reveal dorsal bridge
    concavity, so no unsupported deformation is applied.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must have shape (H, W, 3).")

    if not isinstance(nose_mask, np.ndarray) or nose_mask.ndim != 2:
        raise ValueError("nose_mask must be a 2D NumPy array.")

    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")

    if not np.any(nose_mask):
        raise ValueError("The nose mask is empty.")

    return image.copy()



def hump_nose_deformation(image, nose_mask, strength=0.40, contour_side="left"):
    """
    Apply a mild, localized hump along the nasal dorsum in a profile image.

    This is a 2D contour-warping approximation. Validate results visually.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a nonempty BGR image (H, W, 3).")

    if not isinstance(nose_mask, np.ndarray) or nose_mask.ndim != 2:
        raise ValueError("nose_mask must be a 2D NumPy array.")

    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")

    if contour_side not in ("left", "right"):
        raise ValueError("contour_side must be 'left' or 'right'.")

    h, w = image.shape[:2]

    if nose_mask.shape != (h, w):
        nose_mask = cv2.resize(
            nose_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = (nose_mask > 0).astype(np.uint8)

    if not np.any(mask):
        raise ValueError("The nose mask is empty.")

    ys, xs = np.where(mask > 0)
    y_top, y_bottom = int(ys.min()), int(ys.max())
    x_left, x_right = int(xs.min()), int(xs.max())

    nose_width = x_right - x_left + 1
    nose_height = y_bottom - y_top + 1

    if nose_width < 3 or nose_height < 8:
        raise ValueError("The nose mask is too small for hump deformation.")

    # Extract the visible nasal contour.
    contour = np.full(h, np.nan, dtype=np.float32)

    for y in range(y_top, y_bottom + 1):
        row_x = np.flatnonzero(mask[y])

        if row_x.size:
            contour[y] = (
                float(row_x.min())
                if contour_side == "left"
                else float(row_x.max())
            )

    valid_rows = np.flatnonzero(np.isfinite(contour))

    if valid_rows.size < 5:
        raise ValueError("Not enough valid rows to extract the nose contour.")

    contour = np.interp(
        np.arange(h),
        valid_rows,
        contour[valid_rows],
    ).astype(np.float32)

    contour = cv2.GaussianBlur(
        contour.reshape(-1, 1),
        (0, 0),
        sigmaX=max(1.0, nose_height * 0.012),
    ).ravel()

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Normalized vertical position: root = 0, tip = 1.
    t = np.clip(
        (np.arange(h, dtype=np.float32) - y_top)
        / max(nose_height - 1, 1),
        0.0,
        1.0,
    )

    # Concentrate the hump on the upper-middle dorsum.
    # Narrower width limits deformation near the root and tip.
    hump_center = 0.40
    hump_width = 0.11

    hump_profile = np.exp(
        -0.5 * ((t - hump_center) / hump_width) ** 2
    )

    # Taper movement toward the nasal root and tip.
    endpoint_taper = np.clip(
        np.sin(np.pi * t), 0.0, 1.0
    ) ** 1.5

    profile = (hump_profile * endpoint_taper).astype(np.float32)

    # Signed distance from the visible contour.
    signed_distance = xx - contour[:, None]

    if contour_side == "left":
        face_distance = signed_distance
        direction = -1.0
    else:
        face_distance = -signed_distance
        direction = 1.0

    # Narrower contour band keeps the warp close to the dorsum.
    support_width = max(2.0, nose_width * 0.25)

    contour_band = np.exp(
        -0.5 * (face_distance / support_width) ** 2
    ).astype(np.float32)

    # Reduced displacement for a subtler hump.
    max_shift = strength * nose_width * 0.28

    displacement = (
        direction
        * max_shift
        * profile[:, None]
        * contour_band
    ).astype(np.float32)

    # Feather the boundary to reduce visible seams.
    kernel_size = max(5, int(round(nose_width * 0.55)))

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    expanded_mask = cv2.dilate(mask, kernel)

    support = cv2.GaussianBlur(
        expanded_mask.astype(np.float32),
        (0, 0),
        sigmaX=max(1.0, nose_width * 0.05),
    )

    support = np.clip(support, 0.0, 1.0)
    displacement *= support

    map_x = np.ascontiguousarray(
        xx - displacement,
        dtype=np.float32,
    )
    map_y = np.ascontiguousarray(yy, dtype=np.float32)

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )

    alpha = support[..., None]

    result = (
        warped.astype(np.float32) * alpha
        + image.astype(np.float32) * (1.0 - alpha)
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def frontal_hump_nose(image, nose_mask, strength=0.65):
    """
    Conservative frontal-view approximation of a nasal hump.

    A frontal image cannot directly establish the 3D bridge projection,
    so visual validation is required.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a nonempty BGR image (H, W, 3).")

    if not isinstance(nose_mask, np.ndarray) or nose_mask.ndim != 2:
        raise ValueError("nose_mask must be a 2D NumPy array.")

    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")

    h, w = image.shape[:2]

    if nose_mask.shape != (h, w):
        nose_mask = cv2.resize(
            nose_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = (nose_mask > 0).astype(np.uint8)

    if not np.any(mask):
        raise ValueError("The nose mask is empty.")

    ys, xs = np.where(mask > 0)
    y_top, y_bottom = int(ys.min()), int(ys.max())
    x_left, x_right = int(xs.min()), int(xs.max())

    nose_width = x_right - x_left + 1
    nose_height = y_bottom - y_top + 1

    if nose_width < 3 or nose_height < 8:
        raise ValueError("The nose mask is too small.")

    centerline = np.full(h, np.nan, dtype=np.float32)

    for y in range(y_top, y_bottom + 1):
        row_x = np.flatnonzero(mask[y])

        if row_x.size:
            centerline[y] = (row_x.min() + row_x.max()) / 2.0

    valid_rows = np.flatnonzero(np.isfinite(centerline))

    if valid_rows.size < 5:
        raise ValueError("Unable to estimate the nasal centerline.")

    centerline = np.interp(
        np.arange(h),
        valid_rows,
        centerline[valid_rows],
    ).astype(np.float32)

    centerline = cv2.GaussianBlur(
        centerline.reshape(-1, 1),
        (0, 0),
        sigmaX=max(1.0, nose_height * 0.015),
    ).ravel()

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    t = np.clip(
        (np.arange(h, dtype=np.float32) - y_top)
        / max(nose_height - 1, 1),
        0.0,
        1.0,
    )

    bridge_profile = np.exp(
        -0.5 * ((t - 0.43) / 0.16) ** 2
    )

    endpoint_taper = np.clip(np.sin(np.pi * t), 0.0, 1.0) ** 1.2
    profile = (bridge_profile * endpoint_taper).astype(np.float32)

    horizontal_distance = xx - centerline[:, None]
    sigma = max(2.0, nose_width * 0.35)

    horizontal_profile = np.exp(
        -0.5 * (horizontal_distance / sigma) ** 2
    ).astype(np.float32)

    # This creates a symmetric lateral contour adjustment only.
    # It is not a true reconstruction of a 3D dorsal hump.
    max_shift = strength * nose_width * 0.06

    displacement = (
        max_shift
        * profile[:, None]
        * horizontal_profile
    ).astype(np.float32)

    kernel_size = max(7, int(round(nose_width * 0.7)))

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    expanded = cv2.dilate(mask, kernel)

    support = cv2.GaussianBlur(
        expanded.astype(np.float32),
        (0, 0),
        sigmaX=max(1.0, nose_width * 0.06),
    )

    support = np.clip(support, 0.0, 1.0)
    displacement *= support

    map_x = np.ascontiguousarray(
        xx - np.sign(horizontal_distance) * displacement,
        dtype=np.float32,
    )
    map_y = np.ascontiguousarray(yy, dtype=np.float32)

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )

    alpha = support[..., None]

    result = (
        warped.astype(np.float32) * alpha
        + image.astype(np.float32) * (1.0 - alpha)
    )

    return np.clip(result, 0, 255).astype(np.uint8)

import cv2
import numpy as np


# ============================================================
# EXISTING DEFORMATION FUNCTIONS
# ============================================================
# Your approved saddle-nose and other existing deformation
# functions remain unchanged in the updated file.


def helicopter_hump_nose(image, nose_mask, strength=0.25):
    """
    Handle hump-nose processing for a helicopter (top-down) view.

    A dorsal hump cannot be reliably inferred from a top-down
    image and its 2D nose mask. Therefore, this function
    preserves the original image instead of introducing an
    anatomically misleading deformation.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError(
            "image must be a nonempty BGR image (H, W, 3)."
        )

    if not isinstance(nose_mask, np.ndarray) or nose_mask.ndim != 2:
        raise ValueError("nose_mask must be a 2D NumPy array.")

    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1.")

    if nose_mask.shape != image.shape[:2]:
        nose_mask = cv2.resize(
            nose_mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if not np.any(nose_mask > 0):
        raise ValueError("The nose mask is empty.")

    # Preserve the original image for this view.
    return image.copy()

def basal_hump_nose(image, nose_mask, strength=0.25):
    """
    Preserve the basal view during hump-nose deformation.
    The dorsal hump is not reliably visible from this angle.
    """
    return image.copy()


def deviated_nose_deformation(
    image,
    nose_mask,
    strength=0.25,
    direction="right",
):
    """
    Apply a smooth, localized 2D nasal-deviation approximation.

    Pixels outside the original nose mask remain unchanged.
    This is a 2D image warp, not a 3D anatomical reconstruction.
    """

    # 1. Validate inputs.
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError(
            "image must be a nonempty BGR image with shape (H, W, 3)."
        )

    if (
        not isinstance(nose_mask, np.ndarray)
        or nose_mask.ndim != 2
        or nose_mask.size == 0
    ):
        raise ValueError("nose_mask must be a nonempty 2D array.")

    if not np.isscalar(strength) or not np.isfinite(strength):
        raise ValueError("strength must be a finite number.")

    strength = float(strength)

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0.0 and 1.0.")

    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right'.")

    h, w = image.shape[:2]

    # 2. Resize and binarize the mask.
    if nose_mask.shape != (h, w):
        nose_mask = cv2.resize(
            nose_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = (nose_mask > 0).astype(np.uint8)

    if not np.any(mask):
        raise ValueError("The nose mask is empty.")

    ys, xs = np.where(mask > 0)

    y_top = int(ys.min())
    y_bottom = int(ys.max())
    x_left = int(xs.min())
    x_right = int(xs.max())

    nose_height = y_bottom - y_top + 1
    nose_width = x_right - x_left + 1

    if nose_height < 8 or nose_width < 3:
        raise ValueError("The nose mask is too small for deformation.")

    # 3. Estimate a smooth mask centerline.
    centerline = np.full(h, np.nan, dtype=np.float32)

    for y in range(y_top, y_bottom + 1):
        row_x = np.flatnonzero(mask[y])

        if row_x.size:
            centerline[y] = (
                float(row_x.min()) + float(row_x.max())
            ) / 2.0

    valid_rows = np.flatnonzero(np.isfinite(centerline))

    if valid_rows.size < 5:
        raise ValueError("Unable to estimate the nose centerline.")

    centerline = np.interp(
        np.arange(h),
        valid_rows,
        centerline[valid_rows],
    ).astype(np.float32)

    centerline = cv2.GaussianBlur(
        centerline.reshape(-1, 1),
        (0, 0),
        sigmaX=max(1.0, nose_height * 0.015),
    ).ravel()

    # 4. Create a nonnegative, smoothly tapered vertical profile.
    t = np.clip(
        (
            np.arange(h, dtype=np.float32) - y_top
        ) / max(nose_height - 1, 1),
        0.0,
        1.0,
    )

    sine_profile = np.clip(
        np.sin(np.pi * t),
        0.0,
        1.0,
    )

    vertical_profile = np.power(
        sine_profile,
        1.2,
    ).astype(np.float32)

    vertical_profile[:y_top] = 0.0
    vertical_profile[y_bottom + 1:] = 0.0

    # 5. Smooth the horizontal influence around the nose centerline.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    horizontal_distance = xx - centerline[:, None]

    sigma_x = max(2.0, nose_width * 0.45)

    horizontal_profile = np.exp(
        -0.5 * (horizontal_distance / sigma_x) ** 2
    ).astype(np.float32)

    # 6. Build smooth feathered support and displacement map.
    kernel_size = max(5, int(round(nose_width * 0.15)) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    expanded = cv2.dilate(mask, kernel, iterations=1)

    sigma = max(1.5, nose_width * 0.06)
    support = cv2.GaussianBlur(
        expanded.astype(np.float32),
        (0, 0),
        sigmaX=sigma,
    )
    support = np.clip(support, 0.0, 1.0)

    # Lateral displacement scaling
    sign = 1.0 if direction == "right" else -1.0
    max_displacement = strength * nose_width * 0.48

    displacement = (
        sign
        * max_displacement
        * vertical_profile[:, None]
        * horizontal_profile
        * support
    ).astype(np.float32)

    # 7. Construct reverse-mapping coordinates.
    map_x = np.clip(
        xx - displacement,
        0.0,
        float(w - 1),
    ).astype(np.float32)

    map_y = yy.astype(np.float32)

    # Safety check before remapping.
    if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
        raise ValueError("Non-finite remapping coordinates detected.")

    # 8. Warp the image with cubic interpolation and reflection.
    warped = cv2.remap(
        image,
        np.ascontiguousarray(map_x, dtype=np.float32),
        np.ascontiguousarray(map_y, dtype=np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )

    # 9. Blend the warped image with the original using the feathered support.
    alpha_3d = support[..., None]

    result = (
        warped.astype(np.float32) * alpha_3d
        + image.astype(np.float32) * (1.0 - alpha_3d)
    )

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result




def basal_deviated_nose_deformation(
    image,
    nose_mask,
    strength=0.25,
    direction="right",
):
    """
    Create a visible, localized nasal deviation in a basal view.

    This 2D approximation introduces asymmetric lateral
    displacement while constraining the warp to the nose mask.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a nonempty BGR image (H, W, 3).")

    if (
        not isinstance(nose_mask, np.ndarray)
        or nose_mask.ndim != 2
        or nose_mask.size == 0
    ):
        raise ValueError("nose_mask must be a nonempty 2D array.")

    if not np.isscalar(strength) or not np.isfinite(strength):
        raise ValueError("strength must be a finite number.")

    strength = float(strength)

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0.0 and 1.0.")

    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right'.")

    h, w = image.shape[:2]

    if nose_mask.shape != (h, w):
        nose_mask = cv2.resize(
            nose_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = (nose_mask > 0).astype(np.uint8)

    if not np.any(mask):
        raise ValueError("The nose mask is empty.")

    ys, xs = np.where(mask > 0)

    y_top, y_bottom = int(ys.min()), int(ys.max())
    x_left, x_right = int(xs.min()), int(xs.max())

    nose_height = y_bottom - y_top + 1
    nose_width = x_right - x_left + 1

    if nose_height < 8 or nose_width < 3:
        raise ValueError("The nose mask is too small for deformation.")

    # Build a smooth profile that peaks near the middle of the nose.
    y_positions = np.arange(h, dtype=np.float32)

    t = np.clip(
        (y_positions - y_top) / max(1, nose_height - 1),
        0.0,
        1.0,
    )

    vertical_profile = np.maximum(
        np.sin(np.pi * t), 0.0
    ) ** 1.2

    vertical_profile *= (
        (y_positions >= y_top) & (y_positions <= y_bottom)
    ).astype(np.float32)

    # Expand the nose mask slightly and feather the boundary.
    kernel_size = max(5, int(round(nose_width * 0.15)))

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    expanded = cv2.dilate(mask, kernel)

    support = cv2.GaussianBlur(
        expanded.astype(np.float32),
        (0, 0),
        sigmaX=max(1.0, nose_width * 0.02),
    )

    support = np.clip(support, 0.0, 1.0)

    # Displace the visible nasal region laterally.
    sign = 1.0 if direction == "right" else -1.0

    max_displacement = strength * nose_width * 0.65

    displacement = (
        sign * max_displacement * vertical_profile
    ).astype(np.float32)

    displacement_map = (
        displacement[:, None] * support
    ).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    map_x = np.ascontiguousarray(
        xx - displacement_map,
        dtype=np.float32,
    )

    map_y = np.ascontiguousarray(
        yy,
        dtype=np.float32,
    )

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )

    alpha = support[..., None]

    result = (
        warped.astype(np.float32) * alpha
        + image.astype(np.float32) * (1.0 - alpha)
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def helicopter_deviated_nose_deformation(
    image, nose_mask, strength=0.20, direction="right"
):
    """
    Create a localized lateral deviation of the visible nose in
    helicopter view while preserving the surrounding facial region.

    Args:
        image: Input BGR image.
        nose_mask: Binary nose mask.
        strength: Deformation strength (0 to 1).
        direction: 'right' or 'left'.

    Returns:
        Deformed BGR image.
    """
    import cv2
    import numpy as np

    if image is None or nose_mask is None:
        raise ValueError("Image and nose mask must not be None.")

    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right'.")

    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1.")

    original = image.copy()
    height, width = image.shape[:2]

    if nose_mask.shape[:2] != (height, width):
        nose_mask = cv2.resize(
            nose_mask, (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    if nose_mask.ndim == 3:
        nose_mask = cv2.cvtColor(nose_mask, cv2.COLOR_BGR2GRAY)

    mask = (nose_mask > 0).astype(np.uint8)

    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("The nose mask is empty.")

    x, y, nose_width, nose_height = cv2.boundingRect(points)

    if nose_width < 3 or nose_height < 3:
        raise ValueError("The nose mask is too small to deform.")

    # Build a smooth displacement profile along the nose's length.
    yy, xx = np.mgrid[0:height, 0:width]
    t = np.clip((yy - y) / max(nose_height - 1, 1), 0, 1)

    # Deviation peaks around the middle and tapers toward both ends.
    profile = np.sin(np.pi * t) ** 1.2

    # Extend support slightly beyond the mask and soften the boundary.
    kernel_size = max(3, int(round(nose_width * 0.12)) | 1)
    expanded = cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        ),
        iterations=1
    )

    sigma = max(1.0, nose_width * 0.025)
    support = cv2.GaussianBlur(
        expanded.astype(np.float32), (0, 0), sigmaX=sigma
    )
    support = np.clip(support, 0, 1)

    sign = 1.0 if direction == "right" else -1.0
    max_shift = strength * nose_width * 0.45

    # Localized horizontal displacement; surrounding face stays stable.
    displacement = sign * max_shift * profile * support

    map_x = (xx.astype(np.float32) - displacement).astype(np.float32)
    map_y = yy.astype(np.float32)

    warped = cv2.remap(
        original,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )

    alpha = support[..., None]
    result = (
        warped.astype(np.float32) * alpha
        + original.astype(np.float32) * (1.0 - alpha)
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def _broad_nose_warp(
    image,
    nose_mask,
    strength=0.40,
    view="frontal",
):
    """
    Localized 2D broad-nose approximation using a binary nose mask.

    The nose is widened horizontally around its estimated centerline.
    The deformation is strongest in the lower nose and tapers toward
    the nasal root and tip.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image (H, W, 3).
    nose_mask : np.ndarray
        Binary nose mask (H, W), with nose pixels nonzero.
    strength : float
        Widening strength between 0.0 and 1.0.
        Start with 0.25 and validate visually.
    view : str
        One of: frontal, profile, basal, helicopter.

    Returns
    -------
    np.ndarray
        Widened image with original dimensions.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a nonempty BGR image (H, W, 3).")

    if (
        not isinstance(nose_mask, np.ndarray)
        or nose_mask.ndim not in (2, 3)
        or nose_mask.size == 0
    ):
        raise ValueError("nose_mask must be a nonempty 2D or 3D array.")

    if not np.isscalar(strength) or not np.isfinite(strength):
        raise ValueError("strength must be a finite number.")

    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be between 0.0 and 1.0.")

    if view not in ("frontal", "profile", "basal", "helicopter"):
        raise ValueError(
            "view must be frontal, profile, basal, or helicopter."
        )

    h, w = image.shape[:2]

    # Convert a color mask to grayscale if necessary.
    if nose_mask.ndim == 3:
        nose_mask = cv2.cvtColor(nose_mask, cv2.COLOR_BGR2GRAY)

    if nose_mask.shape != (h, w):
        nose_mask = cv2.resize(
            nose_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = (nose_mask > 0).astype(np.uint8)

    if not np.any(mask):
        raise ValueError("The nose mask is empty.")

    ys, xs = np.where(mask > 0)
    y_top, y_bottom = int(ys.min()), int(ys.max())
    x_left, x_right = int(xs.min()), int(xs.max())

    nose_height = y_bottom - y_top + 1
    nose_width = x_right - x_left + 1

    if nose_height < 3 or nose_width < 3:
        raise ValueError("The nose mask is too small for widening.")

    # Estimate the nose centerline row by row.
    centerline = np.full(h, np.nan, dtype=np.float32)

    for y in range(y_top, y_bottom + 1):
        row_x = np.flatnonzero(mask[y])
        if row_x.size:
            centerline[y] = (row_x.min() + row_x.max()) / 2.0

    valid_rows = np.flatnonzero(np.isfinite(centerline))

    if valid_rows.size < 3:
        raise ValueError("Unable to estimate the nose centerline.")

    centerline = np.interp(
        np.arange(h),
        valid_rows,
        centerline[valid_rows],
    ).astype(np.float32)

    centerline = cv2.GaussianBlur(
        centerline.reshape(-1, 1),
        (0, 0),
        sigmaX=max(1.0, nose_height * 0.012),
    ).ravel()

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Normalized vertical position within the nose.
    t = np.clip(
        (np.arange(h, dtype=np.float32) - y_top)
        / max(nose_height - 1, 1),
        0.0,
        1.0,
    )

    # Emphasize the lower nose while smoothly tapering the effect.
    if view == "frontal":
        lower_center, lower_width = 0.70, 0.25
        view_factor = 1.0
    elif view == "profile":
        lower_center, lower_width = 0.66, 0.23
        view_factor = 0.65
    elif view == "basal":
        # Basal images show the nostrils and nasal base directly.
        lower_center, lower_width = 0.60, 0.30
        view_factor = 0.85
    else:
        # Conservative widening for the top-down view.
        lower_center, lower_width = 0.55, 0.30
        view_factor = 0.60

    vertical_profile = np.exp(
        -0.5 * ((t - lower_center) / lower_width) ** 2
    )

    # Taper the effect at both ends of the visible nose.
    endpoint_taper = np.clip(
        np.sin(np.pi * t), 0.0, 1.0
    ) ** 0.7

    vertical_profile *= endpoint_taper
    vertical_profile *= (
        (np.arange(h) >= y_top) & (np.arange(h) <= y_bottom)
    )

    # Build a feathered support region around the nose.
    kernel_size = max(5, int(round(nose_width * 0.12)) | 1)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    expanded = cv2.dilate(mask, kernel, iterations=1)

    support = cv2.GaussianBlur(
        expanded.astype(np.float32),
        (0, 0),
        sigmaX=max(1.0, nose_width * 0.025),
    )
    support = np.clip(support, 0.0, 1.0)

    # Outward expansion: output pixels sample source pixels closer
    # to the centerline, making the nose appear wider.
    horizontal_distance = xx - centerline[:, None]

    amount = (
        float(strength)
        * view_factor
        * vertical_profile[:, None]
    )

    map_x = (
        centerline[:, None]
        + horizontal_distance / (1.0 + amount)
    ).astype(np.float32)

    map_y = yy.copy()

    # Gradually blend the warp back to the original outside the nose.
    map_x = xx + (map_x - xx) * support

    warped = cv2.remap(
        image,
        np.ascontiguousarray(map_x),
        np.ascontiguousarray(map_y),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )

    alpha = support[..., None]

    result = (
        warped.astype(np.float32) * alpha
        + image.astype(np.float32) * (1.0 - alpha)
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def broad_nose_deformation(image, nose_mask, strength=0.25):
    """General broad-nose deformation for an angled/profile view."""
    return _broad_nose_warp(
        image, nose_mask, strength=strength, view="profile"
    )


def frontal_broad_nose(image, nose_mask, strength=0.25):
    """Widen the lower nose in a frontal view."""
    return _broad_nose_warp(
        image, nose_mask, strength=strength, view="frontal"
    )


def profile_broad_nose(image, nose_mask, strength=0.25):
    """Conservatively widen the visible nose in a profile view."""
    return _broad_nose_warp(
        image, nose_mask, strength=strength, view="profile"
    )


def basal_broad_nose(image, nose_mask, strength=0.25):
    """Widen the nasal base in a basal/underside view."""
    return _broad_nose_warp(
        image, nose_mask, strength=strength, view="basal"
    )


def helicopter_broad_nose(image, nose_mask, strength=0.25):
    """Conservatively widen the visible nose in a helicopter view."""
    return _broad_nose_warp(
        image, nose_mask, strength=strength, view="helicopter"
    )
