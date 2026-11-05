"""
Job service – creation, claiming, state transitions, and progress tracking.

All state-machine transitions go through `transition_job_status()`, which
enforces the permitted transition table defined in JobStatus.

Worker lease / stale-job recovery
----------------------------------
When the worker claims a job it sets `lease_expires_at` to
`now + JOB_LEASE_SECONDS`.  It must call `renew_lease()` regularly.
If the API finds a job whose `lease_expires_at` is in the past but whose
status is still active, it is considered stale and can be re-claimed.
"""

from __future__ import annotations

import logging
import socket
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.exceptions import (
    InvalidJobTransitionException,
    JobNotFoundException,
    JobNotCancellableException,
    JobNotRetryableException,
    RetryLimitExceededException,
)
from backend.app.core.status import DatasetStatus, JobStatus
from backend.app.models.dataset import Dataset
from backend.app.models.generation_job import GenerationJob

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Creation                                                                     #
# --------------------------------------------------------------------------- #

def create_job(
    db: Session,
    *,
    dataset: Dataset,
    provider: str,
    requested_count: int,
) -> GenerationJob:
    """
    Create a new QUEUED generation job for a dataset.

    Does NOT commit – caller is responsible for the transaction boundary.
    """
    job = GenerationJob(
        dataset_id=dataset.id,
        provider=provider,
        status=JobStatus.QUEUED,
        requested_count=requested_count,
        current_stage="Waiting in queue",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    db.add(job)
    return job


# --------------------------------------------------------------------------- #
# Retrieval                                                                    #
# --------------------------------------------------------------------------- #

def get_job_by_id(db: Session, job_id: int) -> GenerationJob | None:
    return db.query(GenerationJob).filter(GenerationJob.id == job_id).first()


def get_jobs_for_dataset(db: Session, dataset_id: int) -> list[GenerationJob]:
    return (
        db.query(GenerationJob)
        .filter(GenerationJob.dataset_id == dataset_id)
        .order_by(GenerationJob.created_at.desc())
        .all()
    )


def get_queued_jobs(db: Session, limit: int = 10) -> list[GenerationJob]:
    """
    Return QUEUED jobs ordered by creation time (oldest first).

    This is the worker's main poll query.  For production use on PostgreSQL,
    replace with SELECT … FOR UPDATE SKIP LOCKED for safety when multiple
    workers run concurrently.
    """
    return (
        db.query(GenerationJob)
        .filter(GenerationJob.status == JobStatus.QUEUED)
        .order_by(GenerationJob.created_at.asc())
        .limit(limit)
        .all()
    )


def get_stale_jobs(db: Session) -> list[GenerationJob]:
    """
    Return active jobs whose lease has expired (worker likely crashed).
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    active_statuses = [
        JobStatus.PREPARING,
        JobStatus.SUBMITTED,
        JobStatus.RUNNING,
        JobStatus.DOWNLOADING,
        JobStatus.VALIDATING,
        JobStatus.PACKAGING,
    ]
    return (
        db.query(GenerationJob)
        .filter(
            GenerationJob.status.in_(active_statuses),
            GenerationJob.lease_expires_at < now,
        )
        .all()
    )


# --------------------------------------------------------------------------- #
# State transitions                                                            #
# --------------------------------------------------------------------------- #

def transition_job_status(
    db: Session,
    job: GenerationJob,
    new_status: JobStatus,
    *,
    stage: str | None = None,
    error_summary: str | None = None,
    commit: bool = True,
) -> GenerationJob:
    """
    Move a job to a new status, enforcing the permitted transition table.

    Parameters
    ----------
    stage         : human-readable stage label (stored in current_stage)
    error_summary : error message stored for FAILED/COMPLETED_WITH_ERRORS
    commit        : whether to commit the session immediately
    """
    current = JobStatus(job.status)

    if not current.can_transition_to(new_status):
        raise InvalidJobTransitionException(
            current=job.status,
            requested=new_status.value,
        )

    job.status = new_status.value
    job.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    if stage is not None:
        job.current_stage = stage

    if error_summary is not None:
        job.error_summary = error_summary

    # Set finished_at for terminal statuses.
    if JobStatus(new_status).is_terminal:
        job.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        job.lease_expires_at = None
        job.worker_id = None

    if commit:
        db.commit()
        db.refresh(job)

    logger.info(
        "Job %d: %s → %s  stage=%s",
        job.id,
        current.value,
        new_status.value,
        job.current_stage,
    )
    return job


# --------------------------------------------------------------------------- #
# Worker lease management                                                      #
# --------------------------------------------------------------------------- #

def claim_job(
    db: Session,
    job: GenerationJob,
    *,
    worker_id: str | None = None,
) -> GenerationJob:
    """
    Claim a QUEUED job for the current worker process.

    Sets the lease and moves the job to PREPARING.
    """
    if worker_id is None:
        worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

    lease_duration = timedelta(seconds=settings.JOB_LEASE_SECONDS)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    job.worker_id = worker_id
    job.lease_expires_at = now + lease_duration
    job.heartbeat_at = now
    job.started_at = job.started_at or now

    return transition_job_status(
        db,
        job,
        JobStatus.PREPARING,
        stage="Preparing generation request",
        commit=True,
    )


def renew_lease(db: Session, job: GenerationJob) -> None:
    """Update the lease expiry and heartbeat timestamp."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    job.heartbeat_at = now
    job.lease_expires_at = now + timedelta(seconds=settings.JOB_LEASE_SECONDS)
    job.updated_at = now
    db.commit()


def requeue_stale_job(db: Session, job: GenerationJob) -> GenerationJob:
    """
    Re-queue a stale job so a healthy worker can pick it up.

    The retry counter is not incremented here – the job failed due to a
    worker crash, not a generation error.
    """
    logger.warning(
        "Requeueing stale job %d (was %s, lease expired at %s)",
        job.id,
        job.status,
        job.lease_expires_at,
    )
    job.worker_id = None
    job.lease_expires_at = None
    job.heartbeat_at = None
    return transition_job_status(
        db,
        job,
        JobStatus.QUEUED,
        stage="Re-queued after stale lease",
        commit=True,
    )


# --------------------------------------------------------------------------- #
# Progress                                                                     #
# --------------------------------------------------------------------------- #

def update_progress(
    db: Session,
    job: GenerationJob,
    *,
    generated: int | None = None,
    downloaded: int | None = None,
    validated: int | None = None,
    accepted: int | None = None,
    rejected: int | None = None,
    failed: int | None = None,
    stage: str | None = None,
    commit: bool = True,
) -> None:
    """Update job counters and recalculate progress_percentage."""
    if generated is not None:
        job.generated_count = generated
    if downloaded is not None:
        job.downloaded_count = downloaded
    if validated is not None:
        job.validated_count = validated
    if accepted is not None:
        job.accepted_count = accepted
    if rejected is not None:
        job.rejected_count = rejected
    if failed is not None:
        job.failed_count = failed
    if stage is not None:
        job.current_stage = stage

    # Percentage is based on accepted images vs. requested.
    if job.requested_count > 0:
        job.progress_percentage = min(
            (job.accepted_count / job.requested_count) * 100.0,
            100.0,
        )
    else:
        job.progress_percentage = 0.0

    job.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    if commit:
        db.commit()


# --------------------------------------------------------------------------- #
# Cancellation                                                                 #
# --------------------------------------------------------------------------- #

def request_cancellation(db: Session, job: GenerationJob) -> GenerationJob:
    """
    Transition an active or queued job to CANCEL_REQUESTED or CANCELLED.

    QUEUED jobs are cancelled immediately.
    Active jobs are moved to CANCEL_REQUESTED so the worker can honour it.
    """
    current = JobStatus(job.status)

    if current.is_terminal:
        raise JobNotCancellableException()

    if current == JobStatus.QUEUED:
        return transition_job_status(db, job, JobStatus.CANCELLED, stage="Cancelled by user")

    if current.is_active:
        return transition_job_status(
            db, job, JobStatus.CANCEL_REQUESTED, stage="Cancellation requested"
        )

    raise JobNotCancellableException()


# --------------------------------------------------------------------------- #
# Retry                                                                        #
# --------------------------------------------------------------------------- #

def retry_job(db: Session, job: GenerationJob) -> GenerationJob:
    """
    Re-queue a FAILED or COMPLETED_WITH_ERRORS job within the retry limit.

    The retry counter is incremented.
    """
    current = JobStatus(job.status)

    if current not in (JobStatus.FAILED, JobStatus.COMPLETED_WITH_ERRORS):
        raise JobNotRetryableException(
            f"Job {job.id} is in status '{job.status}' which is not retryable"
        )

    if job.retry_count >= settings.JOB_MAX_RETRIES:
        raise RetryLimitExceededException(limit=settings.JOB_MAX_RETRIES)

    job.retry_count += 1
    job.error_summary = None
    job.finished_at = None

    return transition_job_status(
        db,
        job,
        JobStatus.QUEUED,
        stage=f"Re-queued for retry #{job.retry_count}",
        commit=True,
    )
