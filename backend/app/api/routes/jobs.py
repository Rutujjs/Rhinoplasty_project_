"""
Generation job endpoints.

Endpoints
---------
GET  /generation-jobs/{job_id}            – job status and progress
GET  /generation-jobs/{job_id}/images     – images associated with the job
POST /generation-jobs/{job_id}/cancel     – request cancellation
POST /generation-jobs/{job_id}/retry      – retry eligible failures
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.exceptions import (
    JobNotFoundException,
    JobNotCancellableException,
    JobNotRetryableException,
    RetryLimitExceededException,
)
from backend.app.core.status import JobStatus
from backend.app.database.session import get_db
from backend.app.models.generated_image import GeneratedImage
from backend.app.schemas.job import JobImageSummary, JobProgressResponse
from backend.app.services.job_service import (
    get_job_by_id,
    request_cancellation,
    retry_job,
)

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/generation-jobs",
    tags=["Generation Jobs"],
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _build_progress_response(job) -> JobProgressResponse:
    remaining = max(0, job.requested_count - job.accepted_count)

    # Generate a human-readable status message.
    status_messages = {
        JobStatus.QUEUED: "Waiting to be processed",
        JobStatus.PREPARING: "Preparing generation request",
        JobStatus.SUBMITTED: "Submitted to generation provider",
        JobStatus.RUNNING: "Running image generation",
        JobStatus.CANCEL_REQUESTED: "Cancellation requested – waiting for provider",
        JobStatus.DOWNLOADING: "Downloading generated images",
        JobStatus.VALIDATING: "Validating images",
        JobStatus.PACKAGING: "Packaging dataset",
        JobStatus.COMPLETED: "Completed successfully",
        JobStatus.COMPLETED_WITH_ERRORS: "Completed with some errors",
        JobStatus.FAILED: "Job failed",
        JobStatus.CANCELLED: "Job was cancelled",
    }
    status_message = status_messages.get(
        JobStatus(job.status),
        job.current_stage or job.status,
    )

    return JobProgressResponse(
        job_id=job.id,
        dataset_id=job.dataset_id,
        provider=job.provider,
        status=job.status,
        current_stage=job.current_stage,
        requested_count=job.requested_count,
        generated_count=job.generated_count,
        downloaded_count=job.downloaded_count,
        validated_count=job.validated_count,
        accepted_count=job.accepted_count,
        rejected_count=job.rejected_count,
        failed_count=job.failed_count,
        progress_percentage=round(job.progress_percentage, 2),
        remaining_count=remaining,
        error_summary=job.error_summary,
        retry_count=job.retry_count,
        provider_run_id=job.provider_run_id,
        created_at=job.created_at,
        started_at=job.started_at,
        updated_at=job.updated_at,
        finished_at=job.finished_at,
        status_message=status_message,
    )


# --------------------------------------------------------------------------- #
# GET /generation-jobs/{job_id}                                                #
# --------------------------------------------------------------------------- #

@router.get(
    "/{job_id}",
    response_model=JobProgressResponse,
    summary="Get generation job status and progress",
)
def get_job_status(
    job_id: int,
    db: Session = Depends(get_db),
):
    job = get_job_by_id(db, job_id)
    if job is None:
        raise JobNotFoundException()

    return _build_progress_response(job)


# --------------------------------------------------------------------------- #
# GET /generation-jobs/{job_id}/images                                         #
# --------------------------------------------------------------------------- #

@router.get(
    "/{job_id}/images",
    response_model=list[JobImageSummary],
    summary="List images produced by this job",
)
def list_job_images(
    job_id: int,
    db: Session = Depends(get_db),
):
    job = get_job_by_id(db, job_id)
    if job is None:
        raise JobNotFoundException()

    images = (
        db.query(GeneratedImage)
        .filter(GeneratedImage.job_id == job_id)
        .order_by(GeneratedImage.id.asc())
        .all()
    )

    return [
        JobImageSummary(
            id=img.id,
            dataset_id=img.dataset_id,
            job_id=img.job_id,
            image_url=f"/images/{img.id}/file",
            validation_status=img.validation_status,
            generation_model=img.generation_model,
            seed=img.seed,
            created_at=img.created_at,
        )
        for img in images
    ]


# --------------------------------------------------------------------------- #
# POST /generation-jobs/{job_id}/cancel                                        #
# --------------------------------------------------------------------------- #

@router.post(
    "/{job_id}/cancel",
    response_model=JobProgressResponse,
    summary="Request job cancellation",
    description=(
        "Queued jobs are cancelled immediately.  Active jobs are moved to "
        "CANCEL_REQUESTED and the worker will attempt provider-side "
        "cancellation.  Completed or already-cancelled jobs return 409."
    ),
)
def cancel_job(
    job_id: int,
    db: Session = Depends(get_db),
):
    job = get_job_by_id(db, job_id)
    if job is None:
        raise JobNotFoundException()

    job = request_cancellation(db, job)
    logger.info("Job %d cancellation requested", job_id)
    return _build_progress_response(job)


# --------------------------------------------------------------------------- #
# POST /generation-jobs/{job_id}/retry                                         #
# --------------------------------------------------------------------------- #

@router.post(
    "/{job_id}/retry",
    response_model=JobProgressResponse,
    summary="Retry a failed or partially-completed job",
    description=(
        "Re-queues a FAILED or COMPLETED_WITH_ERRORS job within the "
        "configured retry limit.  Already-accepted images are preserved."
    ),
)
def retry_job_endpoint(
    job_id: int,
    db: Session = Depends(get_db),
):
    job = get_job_by_id(db, job_id)
    if job is None:
        raise JobNotFoundException()

    job = retry_job(db, job)
    logger.info("Job %d re-queued for retry #%d", job_id, job.retry_count)
    return _build_progress_response(job)
