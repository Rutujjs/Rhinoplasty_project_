"""
Dataset service – CRUD operations for Dataset records.
"""

from sqlalchemy.orm import Session

from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage
from backend.app.models.generation_job import GenerationJob
from backend.app.schemas.dataset import DatasetCreate


def create_dataset(db: Session, dataset_data: DatasetCreate) -> Dataset:
    dataset = Dataset(
        dataset_name=dataset_data.dataset_name,
        region=dataset_data.region,
        population_group=dataset_data.population_group,
        age_min=dataset_data.age_min,
        age_max=dataset_data.age_max,
        sex_or_attribute=dataset_data.sex_or_attribute,
        skin_tone=dataset_data.skin_tone,
        pose=dataset_data.pose,
        expression=dataset_data.expression,
        medical_condition=dataset_data.medical_condition,
        medical_severity=dataset_data.medical_severity,
        num_images=dataset_data.num_images,
        resolution=dataset_data.resolution,
        patient_id=dataset_data.patient_id,
        custom_prompt=dataset_data.custom_prompt,
        viewpoint_prompts=dataset_data.viewpoint_prompts,
        status="CREATED",
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def get_all_datasets(
    db: Session,
    skip: int = 0,
    limit: int = 100,
) -> list[Dataset]:
    return (
        db.query(Dataset)
        .order_by(Dataset.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_dataset_by_id(
    db: Session,
    dataset_id: int,
) -> Dataset | None:
    return db.query(Dataset).filter(Dataset.id == dataset_id).first()


def delete_dataset(db: Session, dataset: Dataset) -> None:
    """
    Delete a dataset and its associated image records.

    Status check (GENERATING guard) is enforced in the route layer.
    This function removes DB records only – files on disk are NOT deleted
    automatically to prevent accidental data loss.
    """
    # Delete child GenerationJob records.
    db.query(GenerationJob).filter(
        GenerationJob.dataset_id == dataset.id
    ).delete(synchronize_session=False)

    # Delete child GeneratedImage records.
    db.query(GeneratedImage).filter(
        GeneratedImage.dataset_id == dataset.id
    ).delete(synchronize_session=False)

    db.delete(dataset)
    db.commit()