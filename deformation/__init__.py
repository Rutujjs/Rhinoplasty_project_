"""
Nasal Deformation Package.

Integrates 2D mask-guided morphology deformation (saddle, hump, broad, deviated)
across all 9 canonical clinical synthetic patient viewpoints.
"""

from deformation.engine import (
    CANONICAL_VIEWPOINTS,
    MORPHOLOGY_METADATA,
    DeformationExecutionResult,
    NasalDeformationEngine,
    deformation_engine,
    normalize_viewpoint_key,
)
from deformation.masking import (
    clean_nose_mask,
    keep_relevant_components,
    load_nose_mask,
    make_overlay,
    prepare_nose_mask,
    save_nose_mask,
)

__all__ = [
    "CANONICAL_VIEWPOINTS",
    "MORPHOLOGY_METADATA",
    "DeformationExecutionResult",
    "NasalDeformationEngine",
    "deformation_engine",
    "normalize_viewpoint_key",
    "clean_nose_mask",
    "keep_relevant_components",
    "load_nose_mask",
    "make_overlay",
    "prepare_nose_mask",
    "save_nose_mask",
]

