from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.dataset import DatasetCreate, DatasetResponse
from app.services.dataset_service import (
    create_dataset,
    get_all_datasets,
    get_dataset_by_id,
)


router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"]
)


@router.post("/", response_model=DatasetResponse)
def create_new_dataset(
    dataset_data: DatasetCreate,
    db: Session = Depends(get_db),
):
    dataset = create_dataset(
        db=db,
        dataset_data=dataset_data,
    )

    return dataset


@router.get("/", response_model=list[DatasetResponse])
def list_datasets(
    db: Session = Depends(get_db),
):
    return get_all_datasets(db)


@router.get("/{dataset_id}", response_model=DatasetResponse)
def retrieve_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = get_dataset_by_id(
        db=db,
        dataset_id=dataset_id,
    )

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found",
        )

    return dataset