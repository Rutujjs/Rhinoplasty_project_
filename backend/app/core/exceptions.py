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


class DatasetNotFoundException(AppException):
    def __init__(self):
        super().__init__(
            message="Dataset not found",
            status_code=404,
        )


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