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
        patient_id=getattr(image, "patient_id", None),
        viewpoint=getattr(image, "viewpoint", None),
        viewpoint_name=getattr(image, "viewpoint_name", None),
        validation_status=getattr(image, "validation_status", None),
        deformation_status=getattr(image, "deformation_status", "NOT_STARTED") or "NOT_STARTED",
        deformed_image_path=getattr(image, "deformed_image_path", None),
        mask_image_path=getattr(image, "mask_image_path", None),
    )
