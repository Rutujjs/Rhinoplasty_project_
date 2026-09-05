from enum import Enum


class DatasetStatus(str, Enum):
    CREATED = "CREATED"
    GENERATING = "GENERATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"