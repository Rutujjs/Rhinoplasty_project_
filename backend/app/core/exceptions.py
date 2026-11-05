"""
Application exception classes.

All exceptions inherit from AppException, which is caught by the FastAPI
exception handler in main.py and converted to a consistent JSON error
response without exposing internal details.
"""


class AppException(Exception):
    """Base exception for application-level errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
    ):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


# ------------------------------------------------------------------ #
# Dataset exceptions                                                   #
# ------------------------------------------------------------------ #

class DatasetNotFoundException(AppException):
    def __init__(self):
        super().__init__(
            message="Dataset not found",
            status_code=404,
        )


class DatasetDeleteForbiddenException(AppException):
    """Raised when trying to delete a dataset that is still being processed."""

    def __init__(self, reason: str = "Dataset cannot be deleted in its current state"):
        super().__init__(
            message=reason,
            status_code=409,
        )


# ------------------------------------------------------------------ #
# Generation / job exceptions (original preserved)                    #
# ------------------------------------------------------------------ #

class GenerationInProgressException(AppException):
    def __init__(self):
        super().__init__(
            message="Dataset generation is already in progress",
            status_code=409,
        )


class DatasetAlreadyCompletedException(AppException):
    def __init__(self):
        super().__init__(
            message="Dataset has already been generated",
            status_code=409,
        )


class GenerationFailedException(AppException):
    def __init__(self):
        super().__init__(
            message="Image generation failed",
            status_code=500,
        )


# ------------------------------------------------------------------ #
# Job exceptions (new)                                                 #
# ------------------------------------------------------------------ #

class JobNotFoundException(AppException):
    def __init__(self):
        super().__init__(
            message="Generation job not found",
            status_code=404,
        )


class InvalidJobTransitionException(AppException):
    """Raised when code tries to move a job to an illegal status."""

    def __init__(self, current: str, requested: str):
        super().__init__(
            message=(
                f"Cannot transition job from '{current}' to '{requested}'"
            ),
            status_code=409,
        )


class JobNotCancellableException(AppException):
    def __init__(self):
        super().__init__(
            message="Job cannot be cancelled in its current state",
            status_code=409,
        )


class JobNotRetryableException(AppException):
    def __init__(self, reason: str = "Job is not eligible for retry"):
        super().__init__(
            message=reason,
            status_code=409,
        )


class RetryLimitExceededException(AppException):
    def __init__(self, limit: int):
        super().__init__(
            message=f"Job has exceeded the maximum retry limit of {limit}",
            status_code=409,
        )


# ------------------------------------------------------------------ #
# Provider / Kaggle exceptions (new)                                  #
# ------------------------------------------------------------------ #

class ProviderNotConfiguredException(AppException):
    def __init__(self, provider: str):
        super().__init__(
            message=f"Generation provider '{provider}' is not configured",
            status_code=500,
        )


class ProviderOperationNotSupportedException(AppException):
    """Raised when a provider does not support a requested operation."""

    def __init__(self, provider: str, operation: str):
        super().__init__(
            message=(
                f"Provider '{provider}' does not support operation '{operation}'"
            ),
            status_code=501,
        )


class KaggleNotConfiguredException(AppException):
    def __init__(self):
        super().__init__(
            message=(
                "Kaggle credentials are not configured. "
                "Set KAGGLE_USERNAME and ensure kaggle.json exists in "
                "KAGGLE_CONFIG_DIR, or set KAGGLE_KEY environment variable."
            ),
            status_code=500,
        )


class KaggleSubmissionFailedException(AppException):
    def __init__(self, detail: str = ""):
        msg = "Kaggle notebook submission failed"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(message=msg, status_code=502)


class KaggleRunFailedException(AppException):
    def __init__(self, detail: str = ""):
        msg = "Kaggle notebook run failed"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(message=msg, status_code=502)


class KaggleTimeoutException(AppException):
    def __init__(self):
        super().__init__(
            message="Kaggle notebook run exceeded the configured timeout",
            status_code=504,
        )


class ArtifactRetrievalException(AppException):
    def __init__(self, detail: str = ""):
        msg = "Failed to retrieve generation output artifacts"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(message=msg, status_code=502)


# ------------------------------------------------------------------ #
# Validation / image exceptions                                        #
# ------------------------------------------------------------------ #

class ImageNotFoundException(AppException):
    def __init__(self):
        super().__init__(
            message="Image not found",
            status_code=404,
        )


class ImageFileNotFoundException(AppException):
    def __init__(self):
        super().__init__(
            message="Image file not found on disk",
            status_code=404,
        )


# ------------------------------------------------------------------ #
# Export exceptions                                                    #
# ------------------------------------------------------------------ #

class ExportNotFoundException(AppException):
    def __init__(self):
        super().__init__(
            message="Export not found or not yet available",
            status_code=404,
        )


class ExportFailedException(AppException):
    def __init__(self, detail: str = ""):
        msg = "Dataset export failed"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(message=msg, status_code=500)