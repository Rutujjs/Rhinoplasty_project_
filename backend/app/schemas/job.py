"""
Pydantic schemas for generation jobs, progress, and dataset export.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Job submission response                                                      #
# --------------------------------------------------------------------------- #

class JobSubmitResponse(BaseModel):
    """
    Returned by POST /datasets/{id}/generate when a job is enqueued.

    HTTP 202 Accepted is used for this response.
    """

    job_id: int = Field(description="Internal identifier for the queued job")
    dataset_id: int
    status: str = Field(description="Initial status – always QUEUED")
    provider: str
    status_url: str = Field(
        description="URL the client should poll for job progress"
    )
    message: str = "Generation job queued successfully"


# --------------------------------------------------------------------------- #
# Job progress / status                                                        #
# --------------------------------------------------------------------------- #

class JobProgressResponse(BaseModel):
    """
    Returned by GET /generation-jobs/{job_id}.

    Fields are a superset of the original GenerationStatusResponse so that
    existing clients polling /datasets/{id}/generation-status continue to
    work (that endpoint is kept for backward compatibility).
    """

    job_id: int
    dataset_id: int
    provider: str

    status: str
    current_stage: str | None

    # Counts
    requested_count: int
    generated_count: int
    downloaded_count: int
    validated_count: int
    accepted_count: int
    rejected_count: int
    failed_count: int

    # Progress
    progress_percentage: float
    remaining_count: int

    # Error information (safe for clients – no stack traces)
    error_summary: str | None

    # Retry information
    retry_count: int

    # Provider reference
    provider_run_id: str | None

    # Timestamps
    created_at: datetime
    started_at: datetime | None
    updated_at: datetime
    finished_at: datetime | None

    # Human-readable status message
    status_message: str

    # Complete Stage and Per-View Timing
    timing: dict | None = None
    estimated_remaining_seconds: float | None = None

    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- #
# Image list for a job                                                         #
# --------------------------------------------------------------------------- #

class JobImageSummary(BaseModel):
    """Minimal image record returned by GET /generation-jobs/{id}/images."""

    id: int
    dataset_id: int
    job_id: int | None
    image_url: str
    validation_status: str
    generation_model: str
    seed: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- #
# Export                                                                       #
# --------------------------------------------------------------------------- #

class ExportRequest(BaseModel):
    """Body for POST /datasets/{id}/export (currently no required fields)."""

    include_rejected: bool = Field(
        default=False,
        description="If true, include rejected images in the ZIP",
    )
    format: str = Field(
        default="zip",
        description="Export format; only 'zip' is supported",
    )


class ExportStatusResponse(BaseModel):
    """
    Returned by GET /datasets/{id}/export.

    When status is READY, download_url will be set.
    When status is PENDING or BUILDING, poll again later.
    """

    dataset_id: int
    status: str = Field(
        description="PENDING | BUILDING | READY | FAILED"
    )
    export_path: str | None = Field(
        default=None,
        description="Server-side file path of the completed ZIP",
    )
    download_url: str | None = Field(
        default=None,
        description="URL to download the ZIP when status is READY",
    )
    file_size_bytes: int | None = None
    created_at: datetime | None = None
    message: str = ""
