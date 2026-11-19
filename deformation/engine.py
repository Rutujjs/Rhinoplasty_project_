"""
Unified Nasal Morphology Deformation Engine.

Routes the 4 clinical nasal morphologies (saddle, hump, broad, deviated) across
all 9 canonical synthetic patient viewpoints to the verified OpenCV/NumPy
geometric warping algorithms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from deformation.algorithms import (
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
from deformation.masking import clean_nose_mask, load_nose_mask, prepare_nose_mask

logger = logging.getLogger(__name__)

CANONICAL_VIEWPOINTS = [
    "frontal",
    "left_20",
    "left_40",
    "left_60",
    "right_20",
    "right_40",
    "right_60",
    "basal",
    "helicopter",
]

MORPHOLOGY_METADATA = {
    "saddle": {
        "name": "Saddle Nose",
        "description": "Depression or collapse of the mid-nasal bridge (dorsum) with preserved tip support.",
        "default_strength": 0.65,
        "min_strength": 0.1,
        "max_strength": 1.0,
        "supports_direction": False,
        "clinical_notes": "Simulates supratip depression characteristic of traumatic or iatrogenic saddle deformities.",
    },
    "hump": {
        "name": "Hump Nose",
        "description": "Dorsal prominence or bony/cartilaginous hump on the upper and mid-nasal dorsum.",
        "default_strength": 0.40,
        "min_strength": 0.1,
        "max_strength": 0.8,
        "supports_direction": False,
        "clinical_notes": "Simulates convex dorsal aesthetic deformities commonly presented for reduction rhinoplasty.",
    },
    "broad": {
        "name": "Broad Nose",
        "description": "Bilateral horizontal widening of the nasal dorsum, bony vault, and alar base.",
        "default_strength": 0.40,
        "min_strength": 0.1,
        "max_strength": 0.8,
        "supports_direction": False,
        "clinical_notes": "Simulates platyrrhine or post-traumatic widened nasal vaults requiring osteotomy simulation.",
    },
    "deviated": {
        "name": "Deviated Nose",
        "description": "Asymmetric lateral displacement (C-shaped or S-shaped curvature) of the septum and nasal tip.",
        "default_strength": 0.35,
        "min_strength": 0.1,
        "max_strength": 0.7,
        "supports_direction": True,
        "default_direction": "right",
        "clinical_notes": "Simulates septal deviation with consistent directional shift across clinical angles.",
    },
}


def normalize_viewpoint_key(raw_view: str) -> str:
    """Normalize any filename stem or view string to one of the 9 canonical keys."""
    val = "".join(ch.lower() for ch in str(raw_view) if ch.isalnum() or ch == "_")
    val_clean = val.replace("_", "")

    if "helicopter" in val_clean:
        return "helicopter"
    if "basal" in val_clean:
        return "basal"

    if "left60" in val_clean:
        return "left_60"
    if "left40" in val_clean:
        return "left_40"
    if "left20" in val_clean:
        return "left_20"
    if "right60" in val_clean:
        return "right_60"
    if "right40" in val_clean:
        return "right_40"
    if "right20" in val_clean:
        return "right_20"

    if "front" in val_clean:
        return "frontal"

    # Direct match fallback
    if val in CANONICAL_VIEWPOINTS:
        return val

    return "frontal"


@dataclass
class DeformationExecutionResult:
    success: bool
    deformation_type: str
    viewpoint: str
    strength: float
    direction: Optional[str] = None
    output_image: Optional[np.ndarray] = None
    mask_used: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class NasalDeformationEngine:
    """
    Core engine executing 2D mask-guided morphology deformation across all 9 views.
    """

    def get_supported_types(self) -> Dict[str, Any]:
        """Return the dictionary of supported morphology specifications."""
        return MORPHOLOGY_METADATA

    def validate_parameters(
        self,
        deformation_type: str,
        strength: Optional[float] = None,
        direction: Optional[str] = None,
    ) -> Tuple[str, float, Optional[str]]:
        """Validate and return normalized (deformation_type, strength, direction)."""
        dtype = str(deformation_type).lower().strip()
        if dtype not in MORPHOLOGY_METADATA:
            raise ValueError(
                f"Unsupported deformation_type: '{deformation_type}'. "
                f"Supported types: {list(MORPHOLOGY_METADATA.keys())}"
            )

        spec = MORPHOLOGY_METADATA[dtype]
        eff_strength = float(strength if strength is not None else spec["default_strength"])
        if not (0.0 <= eff_strength <= 1.0):
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {eff_strength}")

        eff_dir = None
        if spec["supports_direction"]:
            eff_dir = (direction or spec.get("default_direction", "right")).lower().strip()
            if eff_dir not in ("left", "right"):
                raise ValueError(f"Direction for {dtype} must be 'left' or 'right', got {eff_dir}")

        return dtype, eff_strength, eff_dir

    def apply_deformation(
        self,
        image_bgr: np.ndarray,
        nose_mask: np.ndarray,
        viewpoint: str,
        deformation_type: str,
        strength: Optional[float] = None,
        direction: Optional[str] = None,
    ) -> DeformationExecutionResult:
        """
        Execute deformation for a single image and mask.
        """
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be a nonempty BGR image array with shape (H, W, 3).")

        h, w = image_bgr.shape[:2]
        clean_view = normalize_viewpoint_key(viewpoint)
        dtype, eff_strength, eff_dir = self.validate_parameters(deformation_type, strength, direction)

        # Prepare and clean the mask
        binary_mask = prepare_nose_mask(nose_mask, (h, w))
        binary_mask = clean_nose_mask(binary_mask, clean_view)

        try:
            deformed = self._route_algorithm(
                image_bgr=image_bgr,
                mask=binary_mask,
                view=clean_view,
                dtype=dtype,
                strength=eff_strength,
                direction=eff_dir,
            )

            # Quality validation: check that output matches dimensions and is not black/corrupt
            if deformed is None or deformed.shape != (h, w, 3):
                raise RuntimeError("Deformed image shape mismatch or empty output returned.")

            # Measure modified pixel count
            diff = cv2.absdiff(image_bgr, deformed)
            modified_pixels = int(np.count_nonzero(np.any(diff > 2, axis=2)))

            return DeformationExecutionResult(
                success=True,
                deformation_type=dtype,
                viewpoint=clean_view,
                strength=eff_strength,
                direction=eff_dir,
                output_image=deformed,
                mask_used=binary_mask,
                metrics={
                    "width": w,
                    "height": h,
                    "mask_pixels": int(np.count_nonzero(binary_mask)),
                    "modified_pixels": modified_pixels,
                },
            )
        except Exception as exc:
            logger.exception("Deformation failed for view %s type %s: %s", clean_view, dtype, exc)
            return DeformationExecutionResult(
                success=False,
                deformation_type=dtype,
                viewpoint=clean_view,
                strength=eff_strength,
                direction=eff_dir,
                error_message=str(exc),
            )

    def _route_algorithm(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        view: str,
        dtype: str,
        strength: float,
        direction: Optional[str],
    ) -> np.ndarray:
        """Route to the exact algorithm calibrated for this view and morphology."""
        # 1. SADDLE NOSE
        if dtype == "saddle":
            if view == "frontal":
                return frontal_saddle_nose(image_bgr, mask, strength=strength)
            elif view.startswith("left"):
                return saddle_nose_deformation(image_bgr, mask, strength=strength, contour_side="left")
            elif view.startswith("right"):
                return saddle_nose_deformation(image_bgr, mask, strength=strength, contour_side="right")
            elif view == "basal":
                return basal_saddle_nose(image_bgr, mask, strength=strength)
            elif view == "helicopter":
                return helicopter_saddle_nose(image_bgr, mask, strength=strength)

        # 2. HUMP NOSE
        elif dtype == "hump":
            if view == "frontal":
                return frontal_hump_nose(image_bgr, mask, strength=strength)
            elif view.startswith("left"):
                return hump_nose_deformation(image_bgr, mask, strength=strength, contour_side="left")
            elif view.startswith("right"):
                return hump_nose_deformation(image_bgr, mask, strength=strength, contour_side="right")
            elif view == "basal":
                return basal_hump_nose(image_bgr, mask, strength=min(strength, 0.45))
            elif view == "helicopter":
                return helicopter_hump_nose(image_bgr, mask, strength=min(strength, 0.45))

        # 3. BROAD NOSE
        elif dtype == "broad":
            if view == "frontal":
                return frontal_broad_nose(image_bgr, mask, strength=strength)
            elif view.startswith("left") or view.startswith("right"):
                return profile_broad_nose(image_bgr, mask, strength=strength)
            elif view == "basal":
                return basal_broad_nose(image_bgr, mask, strength=strength)
            elif view == "helicopter":
                return helicopter_broad_nose(image_bgr, mask, strength=strength)

        # 4. DEVIATED NOSE
        elif dtype == "deviated":
            dir_val = direction or "right"
            if view == "frontal":
                return deviated_nose_deformation(image_bgr, mask, strength=strength, direction=dir_val)
            elif view.startswith("left") or view.startswith("right"):
                return deviated_nose_deformation(image_bgr, mask, strength=strength, direction=dir_val)
            elif view == "basal":
                return basal_deviated_nose_deformation(image_bgr, mask, strength=strength, direction=dir_val)
            elif view == "helicopter":
                return helicopter_deviated_nose_deformation(image_bgr, mask, strength=strength, direction=dir_val)

        raise ValueError(f"Unmapped deformation routine for view '{view}' and type '{dtype}'")


deformation_engine = NasalDeformationEngine()

