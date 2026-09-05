from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.base import Base


class GeneratedImage(Base):
    __tablename__ = "generated_images"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id"),
        nullable=False,
        index=True
    )

    image_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False
    )

    generation_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    seed: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    prompt: Mapped[str] = mapped_column(
        String(2000),
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="GENERATED"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )