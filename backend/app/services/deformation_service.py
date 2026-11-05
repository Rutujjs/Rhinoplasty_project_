"""
Deformation Service Interface & Integration Boundary.

The nose/facial deformation backend is developed in parallel by another team member.
This service provides clean architecture contracts and placeholder dispatchers so that
when the deformation algorithms (mesh warping, mask generation, inpainting) are ready,
they can be integrated with zero disruption to the generation architecture.

Architectural Guarantees:
  1. Original images are NEVER mutated or overwritten.
  2. Fake deformed images are NEVER generated in place of real algorithms.
  3. Processing states explicitly report PENDING_INTEGRATION / NOT_STARTED.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from backend.app.core.exceptions import AppException
from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage
from backend.app.schemas.deformation import (
    DeformationParameters,
    DeformationStatus,
    DeformationStatusResponse,
    ViewpointDeformationSummary,
)
from generation.viewpoints import CLINICAL_VIEWPOINTS, VIEWPOINT_MAP

logger = logging.getLogger(__name__)


class DeformationBackendInterface(ABC):
    """
    Abstract contract for the incoming deformation engine.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if deformation model weights and pipeline are installed."""
        ...

    @abstractmethod
    def process_patient_viewpoints(
        self,
        db: Session,
        dataset: Dataset,
        patient_id: str,
        parameters: DeformationParameters,
    ) -> DeformationStatusResponse:
        """
        Execute deformation, mask creation, and inpainting on patient viewpoint images.
        """
        ...


class PlaceholderDeformationService(DeformationBackendInterface):
    """
    Current active implementation before the real deformation backend files arrive.
    Accurately tracks status without fake outputs.
    """

    def is_available(self) -> bool:
        return False

    def process_patient_viewpoints(
        self,
        db: Session,
        dataset: Dataset,
        patient_id: str,
        parameters: DeformationParameters,
    ) -> DeformationStatusResponse:
        logger.info(
            "Deformation requested for dataset %d (patient %s), but deformation backend is PENDING_INTEGRATION.",
            dataset.id,
            patient_id,
        )
        return self.get_dataset_deformation_status(db, dataset.id, patient_id)

    def get_dataset_deformation_status(
        self,
        db: Session,
        dataset_id: int,
        patient_id: Optional[str] = None,
    ) -> DeformationStatusResponse:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise AppException(message=f"Dataset {dataset_id} not found", status_code=404)

        effective_pid = patient_id or dataset.patient_id or f"P{dataset.id:02d}"

        # Fetch original generated images for this dataset
        images = (
            db.query(GeneratedImage)
            .filter(GeneratedImage.dataset_id == dataset_id)
            .all()
        )

        viewpoint_summaries: List[ViewpointDeformationSummary] = []
        for img in images:
            vk = img.viewpoint or "frontal"
            v_name = img.viewpoint_name or VIEWPOINT_MAP.get(vk, CLINICAL_VIEWPOINTS[0]).name
            v_status = DeformationStatus(img.deformation_status or "NOT_STARTED")

            viewpoint_summaries.append(
                ViewpointDeformationSummary(
                    viewpoint=vk,
                    viewpoint_name=v_name,
                    original_image_id=img.id,
                    original_image_url=f"/datasets/{dataset_id}/images/{img.id}",
                    deformation_status=v_status,
                    deformed_image_url=img.deformed_image_path,
                    mask_image_url=img.mask_image_path,
                    processing_notes="Deformation backend is currently pending integration by team member.",
                    version=1,
                )
            )

        overall_status = DeformationStatus.PENDING_INTEGRATION if images else DeformationStatus.NOT_STARTED

        return DeformationStatusResponse(
            dataset_id=dataset_id,
            patient_id=effective_pid,
            status=overall_status,
            backend_integration_ready=False,
            message=(
                "Deformation service interface is initialized and ready. "
                "Awaiting arrival of deformation backend pipeline files."
            ),
            viewpoints=viewpoint_summaries,
        )


deformation_service = PlaceholderDeformationService()
