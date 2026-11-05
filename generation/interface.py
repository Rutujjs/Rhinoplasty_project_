"""
FaceGenerator – base interface for all image generation providers.

Phase 1 baseline: only `generate()` is required.
Phase 2 extension: provider-level operations (submit, poll, retrieve,
cancel) are added as concrete methods that return
ProviderOperationResult.UNSUPPORTED by default.

Providers that do not support a particular operation must return the
explicit UNSUPPORTED result rather than raising NotImplementedError,
so callers can distinguish between a runtime error and a feature gap.

Usage
-----
1.  Implement `generate()` for synchronous providers (MOCK, local GPU).
2.  For batch providers (KAGGLE_BATCH), override the async-style methods
    (submit_batch, poll_status, retrieve_artifacts, request_cancellation).
    The worker process calls these methods – not `generate()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from backend.app.schemas.generation import (
    GenerationRequest,
    GeneratedImageResult,
)
from generation.manifest import GenerationManifest


# --------------------------------------------------------------------------- #
# Provider operation result                                                    #
# --------------------------------------------------------------------------- #

class ProviderOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    PENDING = "PENDING"         # remote job is still running
    UNSUPPORTED = "UNSUPPORTED" # operation not available for this provider
    TRANSIENT_ERROR = "TRANSIENT_ERROR"  # retry is appropriate
    PERMANENT_ERROR = "PERMANENT_ERROR"  # do not retry; fail the job


@dataclass
class ProviderOperationResult:
    """
    Structured result returned by every provider method.

    Attributes
    ----------
    outcome       : one of the ProviderOutcome values
    provider_run_id : remote identifier (e.g. Kaggle kernel run ID)
    data          : arbitrary provider-specific payload (deserialized JSON, etc.)
    message       : human-readable summary safe for logs and API responses
    error_detail  : internal diagnostic detail (never sent to API clients)
    """

    outcome: ProviderOutcome
    provider_run_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error_detail: str = ""

    # Convenience constructors.

    @classmethod
    def success(
        cls,
        *,
        provider_run_id: str | None = None,
        data: dict[str, Any] | None = None,
        message: str = "OK",
    ) -> "ProviderOperationResult":
        return cls(
            outcome=ProviderOutcome.SUCCESS,
            provider_run_id=provider_run_id,
            data=data or {},
            message=message,
        )

    @classmethod
    def pending(cls, *, message: str = "Job is still running") -> "ProviderOperationResult":
        return cls(outcome=ProviderOutcome.PENDING, message=message)

    @classmethod
    def unsupported(cls, *, operation: str = "", provider: str = "") -> "ProviderOperationResult":
        msg = f"Provider '{provider}' does not support '{operation}'" if provider else "Operation not supported"
        return cls(outcome=ProviderOutcome.UNSUPPORTED, message=msg)

    @classmethod
    def transient_error(cls, *, message: str, detail: str = "") -> "ProviderOperationResult":
        return cls(
            outcome=ProviderOutcome.TRANSIENT_ERROR,
            message=message,
            error_detail=detail,
        )

    @classmethod
    def permanent_error(cls, *, message: str, detail: str = "") -> "ProviderOperationResult":
        return cls(
            outcome=ProviderOutcome.PERMANENT_ERROR,
            message=message,
            error_detail=detail,
        )

    @property
    def succeeded(self) -> bool:
        return self.outcome == ProviderOutcome.SUCCESS

    @property
    def is_pending(self) -> bool:
        return self.outcome == ProviderOutcome.PENDING

    @property
    def is_error(self) -> bool:
        return self.outcome in (
            ProviderOutcome.TRANSIENT_ERROR,
            ProviderOutcome.PERMANENT_ERROR,
        )


# --------------------------------------------------------------------------- #
# Abstract base                                                                #
# --------------------------------------------------------------------------- #

class FaceGenerator(ABC):
    """
    Abstract base class for all face generation providers.

    Required
    --------
    generate()  – must be implemented by every concrete provider.

    Optional (default: return UNSUPPORTED)
    ----------------------------------------
    submit_batch()          – submit a remote batch job
    poll_status()           – check remote job status
    retrieve_artifacts()    – download completed output files
    request_cancellation()  – ask the remote system to stop

    The worker calls the optional methods; the API layer calls generate()
    only for synchronous (non-batch) providers.
    """

    # ------------------------------------------------------------------ #
    # Required                                                             #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GeneratedImageResult:
        """
        Synchronously generate a single image and return its result.

        This method is used by the legacy synchronous route and the MOCK
        provider.  Batch providers (KAGGLE_BATCH) may raise
        ProviderOperationNotSupported if single-image generation is not
        meaningful for them.
        """

    # ------------------------------------------------------------------ #
    # Optional – batch provider operations                                 #
    # Default implementations return UNSUPPORTED.                          #
    # ------------------------------------------------------------------ #

    def prepare_batch(
        self,
        manifest: GenerationManifest,
    ) -> ProviderOperationResult:
        """
        Validate and prepare a generation batch before submission.

        Providers that need to upload input files (e.g. the manifest JSON
        to Kaggle) should do so here and return the result.  The manifest
        object will already have been written to disk by the caller.
        """
        return ProviderOperationResult.unsupported(
            operation="prepare_batch",
            provider=self.__class__.__name__,
        )

    def submit_batch(
        self,
        manifest: GenerationManifest,
    ) -> ProviderOperationResult:
        """
        Submit the prepared batch to the remote provider.

        On success, `provider_run_id` must be set in the returned result
        so it can be stored in the database and used for subsequent polls.
        """
        return ProviderOperationResult.unsupported(
            operation="submit_batch",
            provider=self.__class__.__name__,
        )

    def poll_status(
        self,
        provider_run_id: str,
    ) -> ProviderOperationResult:
        """
        Poll the remote provider for the current status of a submitted job.

        Returns:
          - SUCCESS  : the remote job has completed successfully
          - PENDING  : the remote job is still running
          - TRANSIENT_ERROR : poll failed but retry is appropriate
          - PERMANENT_ERROR : remote job failed; do not retry
        """
        return ProviderOperationResult.unsupported(
            operation="poll_status",
            provider=self.__class__.__name__,
        )

    def retrieve_artifacts(
        self,
        provider_run_id: str,
        destination_dir: str,
        manifest: GenerationManifest,
    ) -> ProviderOperationResult:
        """
        Download completed output artifacts to `destination_dir`.

        `data` in the result should contain:
          - "artifact_paths": list of absolute paths to downloaded files
          - "summary_path"  : path to the batch summary JSON, if present
        """
        return ProviderOperationResult.unsupported(
            operation="retrieve_artifacts",
            provider=self.__class__.__name__,
        )

    def request_cancellation(
        self,
        provider_run_id: str,
    ) -> ProviderOperationResult:
        """
        Ask the remote provider to cancel the running job.

        If cancellation is not supported, return UNSUPPORTED.
        If the provider accepted the cancellation request, return SUCCESS.
        Note: success means the request was accepted, not that the job has
        stopped.  Continue polling until the remote status confirms it.
        """
        return ProviderOperationResult.unsupported(
            operation="request_cancellation",
            provider=self.__class__.__name__,
        )

    # ------------------------------------------------------------------ #
    # Provider metadata                                                    #
    # ------------------------------------------------------------------ #

    @property
    def provider_name(self) -> str:
        """Short identifier for this provider, used in logs and DB records."""
        return self.__class__.__name__