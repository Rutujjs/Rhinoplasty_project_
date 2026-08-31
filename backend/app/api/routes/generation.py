from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.generation import GeneratedImageResponse
from app.services.generation_service import generate_dataset_images
from app.services.dataset_service import get_dataset_by_id


router = APIRouter(
    prefix="/datasets",
    tags=["Generation"]
)


@router.post(
    "/{dataset_id}/generate",
    response_model=list[GeneratedImageResponse]
)
def generate_images(
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

    if dataset.status == "GENERATING":
        raise HTTPException(
            status_code=409,
            detail="Dataset generation is already in progress",
        )

    dataset.status = "GENERATING"
    db.commit()

    try:
        generated_images = generate_dataset_images(
            db=db,
            dataset=dataset,
        )

        dataset.status = "GENERATED"
        db.commit()

        return generated_images

    except Exception as exc:
        dataset.status = "FAILED"
        db.commit()

        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(exc)}",
        )