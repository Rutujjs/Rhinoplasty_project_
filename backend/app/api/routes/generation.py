"""
Generation routes.

This module provides two patterns:
1.  POST /datasets/{id}/generate
        Enqueues a generation job and returns HTTP 202 with the job ID.
        The job is processed asynchronously by the background worker.

2.  GET /datasets/{id}/generation-status  (backward-compatible)
        Returns a simplified status for the most recent job on the dataset.
        Existing clients can continue using this endpoint.

The detailed per-job endpoints live in api/routes/jobs.py.
"""

import logging

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.exceptions import (
    DatasetAlreadyCompletedException,
    DatasetNotFoundException,
    GenerationFailedException,
    GenerationInProgressException,
)
from backend.app.core.status import DatasetStatus, JobStatus
from backend.app.database.session import get_db
from backend.app.models.generation_job import GenerationJob
from backend.app.schemas.generation import GenerationStatusResponse
from backend.app.schemas.job import JobSubmitResponse
from backend.app.services.dataset_service import get_dataset_by_id
from backend.app.services.job_service import (
    create_job,
    get_jobs_for_dataset,
)

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/datasets",
    tags=["Generation"],
)


# --------------------------------------------------------------------------- #
# POST /datasets/{dataset_id}/generate                                         #
# --------------------------------------------------------------------------- #

@router.post(
    "/{dataset_id}/generate",
    status_code=202,
    response_model=JobSubmitResponse,
    summary="Enqueue a generation job",
    description=(
        "Creates a generation job for the dataset and returns HTTP 202 with "
        "the job ID.  The actual image generation is performed asynchronously "
        "by the background worker.  Poll GET /generation-jobs/{job_id} for "
        "progress."
    ),
)
def generate_images(
    dataset_id: int,
    response: Response,
    db: Session = Depends(get_db),
):
    # 1. Fetch dataset
    dataset = get_dataset_by_id(db=db, dataset_id=dataset_id)
    if dataset is None:
        raise DatasetNotFoundException()

    # 2. Guard against duplicate submissions (check dataset and existing active jobs)
    active_job = (
        db.query(GenerationJob)
        .filter(
            GenerationJob.dataset_id == dataset_id,
            GenerationJob.status.in_([
                JobStatus.QUEUED,
                JobStatus.PREPARING,
                JobStatus.SUBMITTED,
                JobStatus.RUNNING,
                JobStatus.CANCEL_REQUESTED,
                JobStatus.DOWNLOADING,
                JobStatus.VALIDATING,
                JobStatus.PACKAGING,
            ]),
        )
        .first()
    )
    if active_job or dataset.status in (
        DatasetStatus.QUEUED,
        DatasetStatus.GENERATING,
        DatasetStatus.PROCESSING,
        DatasetStatus.DEFORMING,
    ):
        raise GenerationInProgressException()

    if dataset.status in (DatasetStatus.COMPLETED, DatasetStatus.COMPLETED_WITH_ERRORS):
        raise DatasetAlreadyCompletedException()

    # 3. Determine provider
    provider_name = settings.effective_provider

    try:
        # 4. Mark dataset as QUEUED (synchronized with initial job status)
        dataset.status = DatasetStatus.QUEUED
        db.flush()   # write but don't commit yet

        # 5. Create a durable job record
        job = create_job(
            db=db,
            dataset=dataset,
            provider=provider_name,
            requested_count=dataset.num_images,
        )
        db.flush()   # get job.id assigned

        db.commit()
        db.refresh(job)

        try:
            from backend.worker import wake_worker
            wake_worker()
        except Exception:
            pass

    except Exception:
        db.rollback()
        logger.exception("Failed to enqueue generation job for dataset %d", dataset_id)
        raise GenerationFailedException()

    logger.info(
        "Enqueued job %d for dataset %d (provider=%s, count=%d)",
        job.id,
        dataset_id,
        provider_name,
        dataset.num_images,
    )

    response.status_code = 202
    return JobSubmitResponse(
        job_id=job.id,
        dataset_id=dataset_id,
        status=job.status,
        provider=provider_name,
        status_url=f"/generation-jobs/{job.id}",
        message=(
            f"Generation job {job.id} queued. "
            f"Poll {'/generation-jobs/' + str(job.id)} for progress."
        ),
    )


# --------------------------------------------------------------------------- #
# GET /datasets/{dataset_id}/generation-status  (backward-compatible)         #
# --------------------------------------------------------------------------- #

@router.get(
    "/{dataset_id}/generation-status",
    response_model=GenerationStatusResponse,
    summary="Dataset generation status (legacy endpoint)",
    description=(
        "Returns simplified generation status for the most recent job on this "
        "dataset.  Kept for backward compatibility – prefer "
        "GET /generation-jobs/{job_id} for detailed progress."
    ),
)
def get_generation_status(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = get_dataset_by_id(db=db, dataset_id=dataset_id)
    if dataset is None:
        raise DatasetNotFoundException()

    # Try to get counts from the most recent job.
    jobs = get_jobs_for_dataset(db=db, dataset_id=dataset_id)
    most_recent_job = jobs[0] if jobs else None

    if most_recent_job:
        generated_count = most_recent_job.accepted_count or dataset.generated_count
        generation_attempts = most_recent_job.retry_count + 1
        requested = most_recent_job.requested_count
    else:
        generated_count = dataset.generated_count
        generation_attempts = dataset.generation_attempts
        requested = dataset.num_images

    if requested == 0:
        progress = 0.0
    else:
        progress = min((generated_count / requested) * 100.0, 100.0)

    return GenerationStatusResponse(
        dataset_id=dataset.id,
        status=dataset.status,
        requested_images=requested,
        generated_count=generated_count,
        generation_attempts=generation_attempts,
        progress_percentage=round(progress, 2),
    )