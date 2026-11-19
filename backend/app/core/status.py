"""
Job and dataset lifecycle status enums.

DatasetStatus  – the state of the researcher-visible dataset record.
JobStatus      – the internal state of a single generation job.

Permitted transitions are documented below each enum.  Code that changes
status must call the helper  JobStatus.can_transition_to()  so invalid
moves are caught early.
"""

from __future__ import annotations

from enum import Enum


class DatasetStatus(str, Enum):
    """High-level state of a dataset record.

    Transitions
    -----------
    CREATED    → QUEUED | GENERATING | FAILED
    QUEUED     → GENERATING | CANCELLED | FAILED
    GENERATING → PROCESSING | DEFORMING | COMPLETED | COMPLETED_WITH_ERRORS | FAILED | CANCELLED
    PROCESSING → DEFORMING | COMPLETED | COMPLETED_WITH_ERRORS | FAILED | CANCELLED
    DEFORMING  → COMPLETED | COMPLETED_WITH_ERRORS | FAILED | CANCELLED
    COMPLETED / COMPLETED_WITH_ERRORS / FAILED / CANCELLED → (terminal)
    """

    CREATED = "CREATED"
    QUEUED = "QUEUED"
    GENERATING = "GENERATING"
    PROCESSING = "PROCESSING"
    DEFORMING = "DEFORMING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


def map_job_status_to_dataset_status(job_status: str | JobStatus, stage: str | None = None) -> DatasetStatus:
    """Map internal generation job status to researcher-facing Dataset status.

    Mapping Rules
    -------------
    CREATED          → CREATED
    QUEUED           → QUEUED
    PREPARING        → GENERATING
    SUBMITTED        → GENERATING
    RUNNING          → GENERATING
    DOWNLOADING      → PROCESSING
    VALIDATING       → PROCESSING
    PACKAGING        → DEFORMING (if stage indicates deformation) else PROCESSING
    COMPLETED        → COMPLETED
    COMPLETED_WITH_ERRORS → COMPLETED_WITH_ERRORS
    FAILED           → FAILED
    CANCEL_REQUESTED → CANCELLED
    CANCELLED        → CANCELLED
    """
    if isinstance(job_status, str):
        try:
            js = JobStatus(job_status)
        except ValueError:
            return DatasetStatus.CREATED
    else:
        js = job_status

    if js == JobStatus.QUEUED:
        return DatasetStatus.QUEUED
    elif js in (JobStatus.PREPARING, JobStatus.SUBMITTED, JobStatus.RUNNING):
        return DatasetStatus.GENERATING
    elif js in (JobStatus.DOWNLOADING, JobStatus.VALIDATING):
        return DatasetStatus.PROCESSING
    elif js == JobStatus.PACKAGING:
        if stage and "deform" in stage.lower():
            return DatasetStatus.DEFORMING
        return DatasetStatus.PROCESSING
    elif js == JobStatus.COMPLETED:
        return DatasetStatus.COMPLETED
    elif js == JobStatus.COMPLETED_WITH_ERRORS:
        return DatasetStatus.COMPLETED_WITH_ERRORS
    elif js == JobStatus.FAILED:
        return DatasetStatus.FAILED
    elif js in (JobStatus.CANCELLED, JobStatus.CANCEL_REQUESTED):
        return DatasetStatus.CANCELLED

    return DatasetStatus.CREATED


class JobStatus(str, Enum):
    """Detailed lifecycle state of a generation job.

    Transitions (→ means "may transition to")
    ------------------------------------------
    QUEUED          → PREPARING | CANCELLED | FAILED
    PREPARING       → SUBMITTED | FAILED | CANCELLED
    SUBMITTED       → RUNNING | FAILED | CANCELLED
    RUNNING         → DOWNLOADING | FAILED | CANCELLED | CANCEL_REQUESTED
    CANCEL_REQUESTED→ CANCELLED | RUNNING (if cancel not supported)
    DOWNLOADING     → VALIDATING | FAILED | CANCELLED
    VALIDATING      → PACKAGING | FAILED | CANCELLED
    PACKAGING       → COMPLETED | COMPLETED_WITH_ERRORS | FAILED
    COMPLETED       → (terminal)
    COMPLETED_WITH_ERRORS → (terminal, but retry is allowed)
    FAILED          → QUEUED  (when retried within retry limit)
    CANCELLED       → (terminal)
    """

    QUEUED = "QUEUED"
    PREPARING = "PREPARING"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    DOWNLOADING = "DOWNLOADING"
    VALIDATING = "VALIDATING"
    PACKAGING = "PACKAGING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    # ------------------------------------------------------------------ #
    # Permitted transition map                                             #
    # ------------------------------------------------------------------ #
    _TRANSITIONS: dict  # declared below after all variants are defined

    def can_transition_to(self, next_status: "JobStatus") -> bool:
        """Return True if this → next_status is a permitted transition."""
        allowed = _JOB_TRANSITIONS.get(self, set())
        return next_status in allowed

    @property
    def is_terminal(self) -> bool:
        """Return True if no further status change is expected."""
        return self in _TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        """Return True if the job is actively being worked on."""
        return self in _ACTIVE_STATUSES


# Keep these as module-level constants so JobStatus methods can reference them.
_JOB_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {
        JobStatus.PREPARING,
        JobStatus.CANCELLED,
        JobStatus.FAILED,
    },
    JobStatus.PREPARING: {
        JobStatus.SUBMITTED,
        JobStatus.RUNNING,   # MOCK provider skips SUBMITTED
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.CANCEL_REQUESTED,
        JobStatus.QUEUED,    # Allow recovery of stale lease during PREPARING
    },
    JobStatus.SUBMITTED: {
        JobStatus.RUNNING,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.CANCEL_REQUESTED,
    },
    JobStatus.RUNNING: {
        JobStatus.DOWNLOADING,
        JobStatus.COMPLETED,             # MOCK provider completes directly
        JobStatus.COMPLETED_WITH_ERRORS, # MOCK provider with some failures
        JobStatus.CANCEL_REQUESTED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.CANCEL_REQUESTED: {
        JobStatus.CANCELLED,
        JobStatus.RUNNING,   # if provider does not support cancellation
    },
    JobStatus.DOWNLOADING: {
        JobStatus.VALIDATING,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.VALIDATING: {
        JobStatus.PACKAGING,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.PACKAGING: {
        JobStatus.COMPLETED,
        JobStatus.COMPLETED_WITH_ERRORS,
        JobStatus.FAILED,
    },
    # Terminal states have no outgoing transitions.
    JobStatus.COMPLETED: set(),
    JobStatus.COMPLETED_WITH_ERRORS: set(),
    JobStatus.FAILED: {
        # Allow re-queuing within retry limit.
        JobStatus.QUEUED,
    },
    JobStatus.CANCELLED: set(),
}

_TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.COMPLETED_WITH_ERRORS,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
)

_ACTIVE_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.PREPARING,
        JobStatus.SUBMITTED,
        JobStatus.RUNNING,
        JobStatus.CANCEL_REQUESTED,
        JobStatus.DOWNLOADING,
        JobStatus.VALIDATING,
        JobStatus.PACKAGING,
    }
)


class ImageValidationStatus(str, Enum):
    """Per-image validation outcome stored in the database."""

    PENDING = "PENDING"          # not yet validated
    ACCEPTED = "ACCEPTED"        # passed all checks
    REJECTED = "REJECTED"        # failed at least one check
    NOT_EVALUATED = "NOT_EVALUATED"  # validator unavailable or skipped