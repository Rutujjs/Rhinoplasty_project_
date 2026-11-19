"""
GenerationJob – durable record for a single image-generation job.

One GenerationJob is created for every POST /datasets/{id}/generate call.
The job is processed by the background worker and survives API restarts.

Column notes
------------
provider_run_id   : remote identifier returned by the provider (e.g. Kaggle
                    kernel run ID).  NULL until the job is submitted.
lease_expires_at  : set by the worker when it claims the job.  If the worker
                    crashes, another worker can re-claim after this time.
heartbeat_at      : updated periodically by the active worker so liveness can
                    be assessed even before the lease expires.
batch_id          : identifier for the current in-progress batch within a job.
error_summary     : brief human-readable description of the last error,
                    safe to expose to API clients (no internal tracebacks).
worker_id         : identifies which worker process claimed the job, for
                    diagnostics.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base
from backend.app.core.status import JobStatus


class GenerationJob(Base):
    __tablename__ = "generation_jobs"

    # ------------------------------------------------------------------ #
    # Identity                                                             #
    # ------------------------------------------------------------------ #
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------------ #
    # Provider information                                                 #
    # ------------------------------------------------------------------ #
    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Provider name, e.g. MOCK or KAGGLE_BATCH",
    )

    provider_run_id: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Remote run/kernel ID returned by the provider",
    )

    # ------------------------------------------------------------------ #
    # Status and lifecycle                                                 #
    # ------------------------------------------------------------------ #
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default=JobStatus.QUEUED,
        index=True,
        comment="One of the JobStatus enum values",
    )

    current_stage: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Human-readable description of the current processing stage",
    )

    # ------------------------------------------------------------------ #
    # Counts                                                               #
    # ------------------------------------------------------------------ #
    requested_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Total images requested for this job",
    )

    generated_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Images produced by the remote provider",
    )

    downloaded_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Images successfully downloaded from the provider",
    )

    validated_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Images that have gone through the validation pipeline",
    )

    accepted_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Images that passed all validation checks",
    )

    rejected_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Images that failed at least one validation check",
    )

    failed_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Images that could not be generated or downloaded",
    )

    # ------------------------------------------------------------------ #
    # Progress                                                             #
    # ------------------------------------------------------------------ #
    progress_percentage: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Overall completion percentage 0–100",
    )

    # ------------------------------------------------------------------ #
    # Error tracking                                                       #
    # ------------------------------------------------------------------ #
    error_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Last error message, safe for API clients",
    )

    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="How many times this job has been retried",
    )

    stage_timing: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment="JSON record of durations and timestamps for each processing stage and viewpoint",
    )

    # ------------------------------------------------------------------ #
    # Worker lease / heartbeat                                             #
    # ------------------------------------------------------------------ #
    worker_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Identifies the worker process that claimed this job",
    )

    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        comment="Worker must renew before this time or job becomes re-claimable",
    )

    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        comment="Last time the active worker sent a heartbeat",
    )

    # ------------------------------------------------------------------ #
    # Batch tracking                                                       #
    # ------------------------------------------------------------------ #
    batch_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Identifier of the current or last submitted batch",
    )

    # ------------------------------------------------------------------ #
    # Timestamps                                                           #
    # ------------------------------------------------------------------ #
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        comment="When the worker first claimed this job",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        comment="When the job reached a terminal status",
    )
