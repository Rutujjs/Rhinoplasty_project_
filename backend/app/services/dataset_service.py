from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.schemas.dataset import DatasetCreate


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
        status="CREATED",
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return dataset


def get_all_datasets(db: Session) -> list[Dataset]:
    return db.query(Dataset).order_by(Dataset.created_at.desc()).all()


def get_dataset_by_id(
    db: Session,
    dataset_id: int
) -> Dataset | None:
    return db.query(Dataset).filter(
        Dataset.id == dataset_id
    ).first()