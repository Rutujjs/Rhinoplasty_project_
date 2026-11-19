"""
Pydantic Schemas for 2D Nasal Morphology Deformation Pipeline.

Defines API data contracts for:
  - Morphology types metadata and parameters
  - Single-view deformation execution
  - All-9-viewpoints batch deformation execution
  - Deformation results, comparison, and status tracking
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class DeformationType(str, Enum):
    SADDLE = "saddle"
    HUMP = "hump"
    BROAD = "broad"
    DEVIATED = "deviated"


class DeformationStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    NO_VISIBLE_CHANGE = "NO_VISIBLE_CHANGE"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    PENDING_INTEGRATION = "PENDING_INTEGRATION"  # legacy compatibility


class MorphologyTypeInfo(BaseModel):
    type_key: str
    name: str
    description: str
    default_strength: float
    min_strength: float
    max_strength: float
    supports_direction: bool = False
    default_direction: Optional[str] = None
    clinical_notes: str


class MorphologyTypesResponse(BaseModel):
    morphologies: List[MorphologyTypeInfo]


# --------------------------------------------------------------------------- #
# Request Payloads                                                            #
# --------------------------------------------------------------------------- #

class SingleDeformationRequest(BaseModel):
    """
    Request payload to apply a selected nose deformation to one specific viewpoint.
    """
    image_id: Optional[int] = Field(default=None, description="Database ID of the image to deform")
    viewpoint: Optional[str] = Field(default=None, description="Viewpoint key (e.g. frontal, left_20, etc.)")
    deformation_type: str = Field(description="One of: saddle, hump, broad, deviated")
    strength: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Deformation intensity (0.0 to 1.0)")
    direction: Optional[str] = Field(default=None, description="For deviated nose: 'left' or 'right'")


class BatchDeformationRequest(BaseModel):
    """
    Request payload to apply a selected nose deformation to all nine viewpoints for an identity.
    """
    patient_id: Optional[str] = Field(default=None, description="Patient/Identity ID (e.g. P01)")
    deformation_type: str = Field(description="One of: saddle, hump, broad, deviated")
    strength: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Deformation intensity (0.0 to 1.0)")
    direction: Optional[str] = Field(default=None, description="For deviated nose: 'left' or 'right'")
    views: Optional[Union[List[str], str]] = Field(default="all", description="'all' or list of viewpoint keys")


# --------------------------------------------------------------------------- #
# Legacy Compatibility Request Payloads                                       #
# --------------------------------------------------------------------------- #

class DeformationParameters(BaseModel):
    hump_reduction_mm: Optional[float] = Field(default=None)
    tip_rotation_degrees: Optional[float] = Field(default=None)
    tip_projection_ratio: Optional[float] = Field(default=None)
    alar_base_narrowing_mm: Optional[float] = Field(default=None)
    radix_augmentation_mm: Optional[float] = Field(default=None)
    target_viewpoints: List[str] = Field(default_factory=list)
    extra_options: Dict[str, Any] = Field(default_factory=dict)


class DeformationTriggerRequest(BaseModel):
    patient_id: Optional[str] = None
    deformation_type: Optional[str] = Field(default="saddle")
    strength: Optional[float] = Field(default=None)
    direction: Optional[str] = Field(default=None)
    parameters: Optional[DeformationParameters] = Field(default_factory=DeformationParameters)


# --------------------------------------------------------------------------- #
# Response Payloads                                                           #
# --------------------------------------------------------------------------- #

class ViewpointDeformationSummary(BaseModel):
    viewpoint: str
    viewpoint_name: str
    original_image_id: int
    original_image_url: str
    deformation_status: DeformationStatus
    deformation_type: Optional[str] = None
    strength: Optional[float] = None
    direction: Optional[str] = None
    deformed_image_url: Optional[str] = None
    mask_image_url: Optional[str] = None
    processing_notes: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    version: int = 1


class SingleDeformationResponse(BaseModel):
    success: bool
    dataset_id: int
    image_id: int
    patient_id: str
    viewpoint: str
    viewpoint_name: str
    deformation_type: str
    strength: float
    direction: Optional[str] = None
    original_image_url: str
    deformed_image_url: str
    mask_image_url: Optional[str] = None
    message: str
    metrics: Dict[str, Any] = Field(default_factory=dict)


class BatchDeformationResponse(BaseModel):
    dataset_id: int
    patient_id: str
    deformation_type: str
    strength: float
    direction: Optional[str] = None
    total_requested: int
    succeeded_count: int
    failed_count: int
    status: DeformationStatus
    message: str
    viewpoints: List[ViewpointDeformationSummary] = Field(default_factory=list)


class DeformationStatusResponse(BaseModel):
    dataset_id: int
    patient_id: str
    status: DeformationStatus
    backend_integration_ready: bool = True
    message: str
    active_deformation_type: Optional[str] = None
    viewpoints: List[ViewpointDeformationSummary] = Field(default_factory=list)
    updated_at: Optional[datetime] = None
