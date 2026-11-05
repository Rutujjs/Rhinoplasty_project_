"""
GeneratedImage – stores metadata for every image produced by a generation job.

Original columns (id, dataset_id, image_path, generation_model, seed, prompt,
status, created_at) are kept intact so existing data and queries are unaffected.

New columns added in Phase 1
----------------------------
job_id              : FK to generation_jobs.id (nullable for backward compat)
batch_id            : which batch within the job produced this image
image_id_remote     : the image's unique ID as recorded by the provider
validation_status   : PENDING | ACCEPTED | REJECTED | NOT_EVALUATED
rejection_reasons   : JSON-serialised list of rejection reason strings
image_width         : actual width of the stored file in pixels
image_height        : actual height of the stored file in pixels
image_format        : file format (png, jpg, etc.)
file_size_bytes     : file size on disk
checksum_sha256     : SHA-256 of the file for duplicate detection
validated_at        : when validation completed
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base
from backend.app.core.status import ImageValidationStatus


class GeneratedImage(Base):
    __tablename__ = "generated_images"

    # ------------------------------------------------------------------ #
    # Original columns – unchanged                                         #
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

    image_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    generation_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    seed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    prompt: Mapped[str] = mapped_column(
        String(2000),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="GENERATED",
        comment="Legacy generation status field; kept for backward compatibility",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    # ------------------------------------------------------------------ #
    # New columns added in Phase 1                                         #
    # These are all nullable=True so existing rows are not broken.         #
    # ------------------------------------------------------------------ #

    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("generation_jobs.id"),
        nullable=True,
        index=True,
        comment="Generation job that produced this image",
    )

    batch_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Batch within the job that produced this image",
    )

    image_id_remote: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        comment="Unique ID assigned by the provider notebook",
    )

    validation_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default=ImageValidationStatus.PENDING,
        comment="Validation outcome: PENDING | ACCEPTED | REJECTED | NOT_EVALUATED",
    )

    rejection_reasons: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="JSON array of rejection reason strings",
    )

    image_width: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Actual image width in pixels",
    )

    image_height: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Actual image height in pixels",
    )

    image_format: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        comment="Lowercase file format, e.g. png",
    )

    file_size_bytes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="File size in bytes at time of import",
    )

    checksum_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="SHA-256 hex digest for duplicate detection",
    )

    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        comment="Timestamp when validation completed",
    )

    # ------------------------------------------------------------------ #
    # Multi-View & Future Deformation Integration Columns                 #
    # ------------------------------------------------------------------ #
    patient_id: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Synthetic Patient ID (e.g. P01, patient_001)",
    )

    viewpoint: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Viewpoint key (frontal, left_20, left_40, left_60, right_20, right_40, right_60, basal, helicopter)",
    )

    viewpoint_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Human-readable viewpoint title (e.g. Front (0°), Basal)",
    )

    deformation_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="NOT_STARTED",
        comment="Status of future deformation pipeline: NOT_STARTED | PENDING_INTEGRATION | PROCESSING | COMPLETED | FAILED",
    )

    deformed_image_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Filesystem path to the future deformed image once deformation backend is connected",
    )

    mask_image_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Filesystem path to the future deformation mask once deformation backend is connected",
    )