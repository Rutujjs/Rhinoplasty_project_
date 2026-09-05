import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.exceptions import (
    DatasetAlreadyCompletedException,
    DatasetNotFoundException,
    GenerationFailedException,
    GenerationInProgressException,
)
from backend.app.core.status import DatasetStatus
from backend.app.database.session import get_db
from backend.app.schemas.generation import (
    GeneratedImageResponse,
    GenerationStatusResponse,
)
from backend.app.services.dataset_service import get_dataset_by_id
from backend.app.services.generation_service import generate_dataset_images
from backend.app.services.image_service import build_image_response


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/datasets",
    tags=["Generation"],
)


@router.post(
    "/{dataset_id}/generate",
    response_model=list[GeneratedImageResponse],
)
def generate_images(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    # Fetch dataset
    dataset = get_dataset_by_id(
        db=db,
        dataset_id=dataset_id,
    )

    # Dataset does not exist
    if dataset is None:
        raise DatasetNotFoundException()

    # Generation is already running
    if dataset.status == DatasetStatus.GENERATING:
        raise GenerationInProgressException()

    # Dataset has already been completed
    if dataset.status == DatasetStatus.COMPLETED:
        raise DatasetAlreadyCompletedException()

    try:
        # Mark generation as started
        dataset.status = DatasetStatus.GENERATING
        db.commit()

        # Generate the required images
        generated_images = generate_dataset_images(
            db=db,
            dataset=dataset,
        )

        # Mark generation as completed
        dataset.status = DatasetStatus.COMPLETED
        db.commit()

        return [
            build_image_response(image)
            for image in generated_images
        ]

    except Exception:
        # Log the complete internal traceback for developers
        logger.exception(
            "Generation failed for dataset %s",
            dataset_id,
        )

        # Roll back the failed database transaction
        db.rollback()

        try:
            # Count the failed generation attempt.
            # Successful attempts are already counted
            # inside generation_service.py.
            dataset.generation_attempts += 1

            # Mark dataset as failed
            dataset.status = DatasetStatus.FAILED

            db.commit()

        except Exception:
            # If even the failure-status update fails,
            # roll back that transaction as well.
            db.rollback()

            logger.exception(
                "Failed to update dataset %s status to FAILED",
                dataset_id,
            )

        # Do not expose internal exception details
        # to the API client.
        raise GenerationFailedException()


@router.get(
    "/{dataset_id}/generation-status",
    response_model=GenerationStatusResponse,
)
def get_generation_status(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    # Fetch dataset
    dataset = get_dataset_by_id(
        db=db,
        dataset_id=dataset_id,
    )

    # Dataset does not exist
    if dataset is None:
        raise DatasetNotFoundException()

    # Calculate progress
    if dataset.num_images == 0:
        progress = 0.0
    else:
        progress = min(
            (
                dataset.generated_count
                / dataset.num_images
            ) * 100,
            100.0,
        )

    return GenerationStatusResponse(
        dataset_id=dataset.id,
        status=dataset.status,
        requested_images=dataset.num_images,
        generated_count=dataset.generated_count,
        generation_attempts=dataset.generation_attempts,
        progress_percentage=round(progress, 2),
    )