from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    dataset_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    region: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    population_group: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    age_min: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    age_max: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    sex_or_attribute: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )

    skin_tone: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )

    pose: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )

    expression: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )

    medical_condition: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    medical_severity: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True
    )

    num_images: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    resolution: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="CREATED"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )