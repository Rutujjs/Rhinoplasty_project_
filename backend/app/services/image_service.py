from backend.app.schemas.generation import GeneratedImageResponse


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