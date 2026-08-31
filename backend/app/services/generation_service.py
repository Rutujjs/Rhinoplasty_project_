from pathlib import Path

from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.generated_image import GeneratedImage
from generation.mock_generator import generate_mock_image


def build_prompt(dataset: Dataset) -> str:
    return (
        f"Photorealistic synthetic portrait of a fictional person, "
        f"population group {dataset.population_group}, "
        f"age {dataset.age_min} to {dataset.age_max}, "
        f"sex/attribute {dataset.sex_or_attribute}, "
        f"skin tone {dataset.skin_tone}, "
        f"pose {dataset.pose}, "
        f"expression {dataset.expression}, "
        f"region {dataset.region}."
    )


def generate_dataset_images(
    db: Session,
    dataset: Dataset,
) -> list[GeneratedImage]:

    generated_images = []

    prompt = build_prompt(dataset)

    for _ in range(dataset.num_images):

        result = generate_mock_image(
            prompt=prompt,
            resolution=dataset.resolution,
        )

        relative_path = Path(result["image_path"])

        image_record = GeneratedImage(
            dataset_id=dataset.id,
            image_path=str(relative_path),
            generation_model=result["generation_model"],
            seed=result["seed"],
            prompt=result["prompt"],
            status="GENERATED",
        )

        db.add(image_record)
        generated_images.append(image_record)

    db.commit()

    for image in generated_images:
        db.refresh(image)

    return generated_images