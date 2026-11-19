from backend.app.schemas.generation import GeneratedImageResponse


def build_image_response(image) -> GeneratedImageResponse:
    deformed_path = getattr(image, "deformed_image_path", None)
    mask_path = getattr(image, "mask_image_path", None)
    ds_id = getattr(image, "dataset_id", None)

    image_url = f"/datasets/{ds_id}/images/{image.id}" if ds_id else f"/images/{image.id}/file"
    deformed_url = f"/datasets/{ds_id}/deformation/image/{image.id}" if (ds_id and deformed_path) else (f"/images/{image.id}/deformed" if deformed_path else None)
    mask_url = f"/datasets/{ds_id}/deformation/mask/{image.id}" if (ds_id and mask_path) else (f"/images/{image.id}/mask" if mask_path else None)

    return GeneratedImageResponse(
        id=image.id,
        dataset_id=image.dataset_id,
        image_url=image_url,
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
        deformed_image_path=deformed_path,
        mask_image_path=mask_path,
        deformed_image_url=deformed_url,
        mask_image_url=mask_url,
        deformation_type=getattr(image, "deformation_type", None),
        deformation_strength=getattr(image, "deformation_strength", None),
        deformation_direction=getattr(image, "deformation_direction", None),
    )
