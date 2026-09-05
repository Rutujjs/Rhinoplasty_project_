from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.generation import GeneratedImageResponse
from backend.app.services.generation_service import (
    get_dataset_images,
    get_generated_image_by_id,
)
from backend.app.services.image_service import build_image_response

router = APIRouter(
    prefix="/images",
    tags=["Images"],
)


def build_image_response(image) -> GeneratedImageResponse:
    return GeneratedImageResponse(
        id=image.id,
        dataset_id=image.dataset_id,
        image_url=f"/images/{image.id}/file",
        generation_model=image.generation_model,
        seed=image.seed,
        prompt=image.prompt,
        status=image.status,
        created_at=image.created_at,
    )


@router.get(
    "/dataset/{dataset_id}",
    response_model=list[GeneratedImageResponse],
)
def list_dataset_images(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    images = get_dataset_images(
        db=db,
        dataset_id=dataset_id,
    )

    return [
        build_image_response(image)
        for image in images
    ]


@router.get(
    "/{image_id}",
    response_model=GeneratedImageResponse,
)
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

    return build_image_response(image)


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