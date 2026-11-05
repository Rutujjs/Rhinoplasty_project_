"""
Deformation API Routes.

Exposes endpoints for querying deformation status and placeholders for triggering
future nose/facial deformation pipelines.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.deformation import (
    DeformationStatusResponse,
    DeformationTriggerRequest,
)
from backend.app.services.deformation_service import deformation_service

router = APIRouter(
    prefix="/datasets/{dataset_id}/deformation",
    tags=["Deformation Pipeline (Future Integration)"],
)


@router.get(
    "/status",
    response_model=DeformationStatusResponse,
    summary="Get multi-view deformation status for a dataset",
)
def get_deformation_status(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    return deformation_service.get_dataset_deformation_status(db, dataset_id)


@router.post(
    "/trigger",
    response_model=DeformationStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request deformation processing (Interface ready; awaiting backend pipeline)",
)
def trigger_deformation(
    dataset_id: int,
    request: DeformationTriggerRequest,
    db: Session = Depends(get_db),
):
    # Returns PENDING_INTEGRATION status with clear descriptive details
    return deformation_service.get_dataset_deformation_status(
        db, dataset_id, patient_id=request.patient_id
    )
