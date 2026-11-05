"""
Pydantic Schemas for Future Facial / Nose Deformation Pipeline Integration.

These schemas define the data contracts and statuses for when the deformation
backend (developed separately) is plugged into this project.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DeformationStatus(str, Enum):
    """
    Standard lifecycle statuses for the facial/nose deformation pipeline.
    """
    NOT_STARTED = "NOT_STARTED"
    PENDING_INTEGRATION = "PENDING_INTEGRATION"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DeformationParameters(BaseModel):
    """
    Requested clinical nose deformation parameters.
    """
    hump_reduction_mm: Optional[float] = Field(default=None, description="Dorsal hump reduction in mm")
    tip_rotation_degrees: Optional[float] = Field(default=None, description="Nasal tip upward/downward rotation in degrees")
    tip_projection_ratio: Optional[float] = Field(default=None, description="Nasal tip projection adjustment ratio")
    alar_base_narrowing_mm: Optional[float] = Field(default=None, description="Alar base narrowing in mm")
    radix_augmentation_mm: Optional[float] = Field(default=None, description="Nasal radix augmentation in mm")
    target_viewpoints: List[str] = Field(default_factory=list, description="List of viewpoint keys to deform")
    extra_options: Dict[str, Any] = Field(default_factory=dict, description="Additional backend parameters")


class DeformationTriggerRequest(BaseModel):
    """
    API payload to request deformation processing for a dataset or patient.
    """
    patient_id: Optional[str] = Field(default=None, description="Patient ID to deform")
    parameters: DeformationParameters = Field(default_factory=DeformationParameters)


class ViewpointDeformationSummary(BaseModel):
    viewpoint: str
    viewpoint_name: str
    original_image_id: int
    original_image_url: str
    deformation_status: DeformationStatus
    deformed_image_url: Optional[str] = None
    mask_image_url: Optional[str] = None
    processing_notes: Optional[str] = None
    version: int = 1


class DeformationStatusResponse(BaseModel):
    dataset_id: int
    patient_id: str
    status: DeformationStatus
    backend_integration_ready: bool = False
    message: str
    viewpoints: List[ViewpointDeformationSummary] = Field(default_factory=list)
    updated_at: Optional[datetime] = None
