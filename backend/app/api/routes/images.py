from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.generation import GeneratedImageResponse
from app.services.generation_service import (
    get_dataset_images,
    get_generated_image_by_id,
)


router = APIRouter(
    prefix="/images",
    tags=["Images"],
)


@router.get(
    "/dataset/{dataset_id}",
    response_model=list[GeneratedImageResponse],
)
def list_dataset_images(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    return get_dataset_images(
        db=db,
        dataset_id=dataset_id,
    )


@router.get("/{image_id}", response_model=GeneratedImageResponse)
def get_image_details(
    image_id: int,
    db: Session = Depends(get_db),
):
    image = get_generated_image_by_id(
        db=db,
        image_id=image_id,
    )

    if image is None:
        raise HTTPException(
            status_code=404,
            detail="Image not found",
        )

    return image


@router.get("/{image_id}/file")
def get_image_file(
    image_id: int,
    db: Session = Depends(get_db),
):
    image = get_generated_image_by_id(
        db=db,
        image_id=image_id,
    )

    if image is None:
        raise HTTPException(
            status_code=404,
            detail="Image not found",
        )

    image_path = Path(image.image_path)

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Image file not found",
        )

    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=image_path.name,
    )