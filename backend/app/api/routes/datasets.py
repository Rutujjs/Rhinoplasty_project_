from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.dataset import DatasetCreate
from app.services.dataset_service import create_dataset


router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"]
)


@router.post("/")
def create_new_dataset(
    dataset_data: DatasetCreate,
    db: Session = Depends(get_db),
):
    dataset = create_dataset(
        db=db,
        dataset_data=dataset_data,
    )

    return {
        "message": "Dataset created successfully",
        "dataset_id": dataset.id,
        "status": dataset.status,
    }