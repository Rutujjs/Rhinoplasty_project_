"""
Canonical Viewpoint Definitions for Multi-View Synthetic APAC Face Generation.

The system generates NINE standardized clinical viewpoints for one synthetic patient:
  1. Frontal (0°) - Canonical reference image
  2. Left 20°   - Slight oblique rotation toward image-left
  3. Left 40°   - Classic three-quarter oblique rotation toward image-left
  4. Left 60°   - Strong lateral-oblique rotation toward image-left (calibrated angle 95°)
  5. Right 20°  - Slight oblique rotation toward image-right
  6. Right 40°  - Classic three-quarter oblique rotation toward image-right
  7. Right 60°  - Strong lateral-oblique rotation toward image-right (calibrated angle 60°)
  8. Basal      - Inferior subnasal worm's-eye view (camera below nasal tip)
  9. Helicopter - High-angle elevated frontal view (camera above forehead)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ViewpointSpec:
    key: str
    name: str
    angle_degrees: Optional[int]
    nose_toward: str
    camera_position: str
    grid_order: int
    description: str


CLINICAL_VIEWPOINTS: List[ViewpointSpec] = [
    ViewpointSpec(
        key="frontal",
        name="Front (0°)",
        angle_degrees=0,
        nose_toward="center",
        camera_position="Eye-level, straight frontal",
        grid_order=1,
        description="Canonical frontal reference photograph at 0 degrees",
    ),
    ViewpointSpec(
        key="left_20",
        name="Left 20°",
        angle_degrees=20,
        nose_toward="image-left",
        camera_position="Eye-level, patient head rotated 20° left",
        grid_order=2,
        description="Slight 20-degree oblique clinical view toward image-left",
    ),
    ViewpointSpec(
        key="left_40",
        name="Left 40°",
        angle_degrees=40,
        nose_toward="image-left",
        camera_position="Eye-level, patient head rotated 40° left",
        grid_order=3,
        description="Classic 40-degree three-quarter oblique clinical view toward image-left",
    ),
    ViewpointSpec(
        key="left_60",
        name="Left 60°",
        angle_degrees=60,
        nose_toward="image-left",
        camera_position="Eye-level, patient head rotated 60° left",
        grid_order=4,
        description="Strong 60-degree lateral-oblique clinical view toward image-left",
    ),
    ViewpointSpec(
        key="right_20",
        name="Right 20°",
        angle_degrees=20,
        nose_toward="image-right",
        camera_position="Eye-level, patient head rotated 20° right",
        grid_order=5,
        description="Slight 20-degree oblique clinical view toward image-right",
    ),
    ViewpointSpec(
        key="right_40",
        name="Right 40°",
        angle_degrees=40,
        nose_toward="image-right",
        camera_position="Eye-level, patient head rotated 40° right",
        grid_order=6,
        description="Classic 40-degree three-quarter oblique clinical view toward image-right",
    ),
    ViewpointSpec(
        key="right_60",
        name="Right 60°",
        angle_degrees=60,
        nose_toward="image-right",
        camera_position="Eye-level, patient head rotated 60° right",
        grid_order=7,
        description="Strong 60-degree lateral-oblique clinical view toward image-right",
    ),
    ViewpointSpec(
        key="basal",
        name="Basal",
        angle_degrees=None,
        nose_toward="n/a (camera below nose)",
        camera_position="Steep worm's-eye angle below nasal tip looking up",
        grid_order=8,
        description="True inferior basal subnasal view showing nostrils and columella",
    ),
    ViewpointSpec(
        key="helicopter",
        name="Helicopter",
        angle_degrees=None,
        nose_toward="n/a (camera above head)",
        camera_position="Elevated camera above forehead aimed 40° downward",
        grid_order=9,
        description="High-angle elevated frontal view showing upper nasal dorsum and forehead",
    ),
]

VIEWPOINT_KEYS: List[str] = [v.key for v in CLINICAL_VIEWPOINTS]
VIEWPOINT_MAP: Dict[str, ViewpointSpec] = {v.key: v for v in CLINICAL_VIEWPOINTS}

TOTAL_REQUIRED_VIEWPOINTS: int = len(CLINICAL_VIEWPOINTS)  # Exactly 9
