"""
Deformation API Routes.

Exposes endpoints for:
  - Querying available clinical morphology types (/types)
  - Querying multi-view deformation status (/status, /results)
  - Applying single-view deformation (/single)
  - Applying all-9-viewpoints batch deformation (/batch)
  - Serving deformed images (/image/{image_id})
  - Serving segmentation/deformation masks (/mask/{image_id})
  - Legacy trigger endpoint (/trigger)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage
from backend.app.schemas.deformation import (
    BatchDeformationRequest,
    BatchDeformationResponse,
    DeformationStatusResponse,
    DeformationTriggerRequest,
    MorphologyTypesResponse,
    SingleDeformationRequest,
    SingleDeformationResponse,
)
from backend.app.services.deformation_service import deformation_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/datasets/{dataset_id}/deformation",
    tags=["Nasal Morphology Deformation"],
)

# Global router for top-level access without dataset ID
global_deformation_router = APIRouter(
    prefix="/deformation",
    tags=["Nasal Morphology Deformation"],
)


@global_deformation_router.get(
    "/types",
    response_model=MorphologyTypesResponse,
    summary="Get available nasal morphology types and parameters",
)
def get_global_morphology_types() -> MorphologyTypesResponse:
    """Return catalog of available clinical nasal morphology types."""
    return deformation_service.get_morphology_types()


@router.get(
    "/types",
    response_model=MorphologyTypesResponse,
    summary="Get available nasal morphology types for dataset",
)
def get_dataset_morphology_types(dataset_id: int) -> MorphologyTypesResponse:
    """Return catalog of available clinical nasal morphology types."""
    return deformation_service.get_morphology_types()


@router.get(
    "/status",
    response_model=DeformationStatusResponse,
    summary="Get multi-view deformation status for a dataset",
)
def get_deformation_status(
    dataset_id: int,
    patient_id: Optional[str] = None,
    db: Session = Depends(get_db),
) -> DeformationStatusResponse:
    """Query current status of nasal deformations for all viewpoints in the dataset."""
    return deformation_service.get_dataset_deformation_status(db, dataset_id, patient_id=patient_id)


@router.get(
    "/results",
    response_model=DeformationStatusResponse,
    summary="Get multi-view deformation results and summaries",
)
def get_deformation_results(
    dataset_id: int,
    patient_id: Optional[str] = None,
    db: Session = Depends(get_db),
) -> DeformationStatusResponse:
    """Alias for status, returning complete deformation result summaries for all viewpoints."""
    return deformation_service.get_dataset_deformation_status(db, dataset_id, patient_id=patient_id)


@router.post(
    "/single",
    response_model=SingleDeformationResponse,
    summary="Apply selected nasal morphology to a single viewpoint",
)
def apply_single_viewpoint_deformation(
    dataset_id: int,
    request: SingleDeformationRequest,
    db: Session = Depends(get_db),
) -> SingleDeformationResponse:
    """
    Apply a selected nasal deformation (saddle, hump, broad, deviated)
    to a single viewpoint of the dataset's facial identity.
    """
    return deformation_service.process_single_viewpoint(db, dataset_id, request)


@router.post(
    "/batch",
    response_model=BatchDeformationResponse,
    summary="Apply selected nasal morphology to all 9 viewpoints",
)
def apply_batch_deformation(
    dataset_id: int,
    request: BatchDeformationRequest,
    db: Session = Depends(get_db),
) -> BatchDeformationResponse:
    """
    Apply a selected nasal deformation (saddle, hump, broad, deviated)
    across all 9 canonical viewpoints for the dataset's facial identity.
    """
    return deformation_service.process_all_viewpoints(db, dataset_id, request)


@router.post(
    "/trigger",
    response_model=DeformationStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger deformation processing (backward-compatible)",
)
def trigger_deformation(
    dataset_id: int,
    request: DeformationTriggerRequest,
    db: Session = Depends(get_db),
) -> DeformationStatusResponse:
    """
    Legacy trigger endpoint maintained for backward compatibility.
    """
    # If the dataset has images and deformation_type is requested, run batch
    dtype = request.deformation_type or "saddle"
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset:
        images = db.query(GeneratedImage).filter(GeneratedImage.dataset_id == dataset_id).all()
        if images:
            batch_req = BatchDeformationRequest(
                patient_id=request.patient_id,
                deformation_type=dtype,
                strength=request.strength,
                direction=request.direction,
            )
            deformation_service.process_all_viewpoints(db, dataset_id, batch_req)

    return deformation_service.get_dataset_deformation_status(
        db, dataset_id, patient_id=request.patient_id
    )


@router.get(
    "/image/{image_id}",
    summary="Download or stream deformed viewpoint image",
)
def get_deformed_image_file(
    dataset_id: int,
    image_id: int,
    db: Session = Depends(get_db),
):
    """Serve the deformed image PNG for a given generated image record."""
    image = (
        db.query(GeneratedImage)
        .filter(GeneratedImage.id == image_id, GeneratedImage.dataset_id == dataset_id)
        .first()
    )
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found in dataset {dataset_id}",
        )

    if not image.deformed_image_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No deformed image available yet for image {image_id}",
        )

    filepath = Path(image.deformed_image_path)
    if not filepath.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deformed image file missing on disk: {filepath.name}",
        )

    return FileResponse(
        path=str(filepath),
        media_type="image/png",
        filename=filepath.name,
    )


@router.get(
    "/mask/{image_id}",
    summary="Download or stream deformation segmentation mask",
)
def get_deformation_mask_file(
    dataset_id: int,
    image_id: int,
    db: Session = Depends(get_db),
):
    """Serve the segmentation/deformation mask PNG for a given generated image record."""
    image = (
        db.query(GeneratedImage)
        .filter(GeneratedImage.id == image_id, GeneratedImage.dataset_id == dataset_id)
        .first()
    )
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found in dataset {dataset_id}",
        )

    if not image.mask_image_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No mask available for image {image_id}",
        )

    filepath = Path(image.mask_image_path)
    if not filepath.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Mask file missing on disk: {filepath.name}",
        )

    return FileResponse(
        path=str(filepath),
        media_type="image/png",
        filename=filepath.name,
    )
