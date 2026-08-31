from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.models.generated_image import GeneratedImage
from app.schemas.generation import GenerationRequest
from generation.mock_generator import MockFaceGenerator
generator = MockFaceGenerator()

def build_generation_request(
    dataset: Dataset,
) -> GenerationRequest:

    return GenerationRequest(
        age_min=dataset.age_min,
        age_max=dataset.age_max,
        sex_or_attribute=dataset.sex_or_attribute,
        skin_tone=dataset.skin_tone,
        population_group=dataset.population_group,
        region=dataset.region,
        pose=dataset.pose,
        expression=dataset.expression,
        medical_condition=dataset.medical_condition,
        medical_severity=dataset.medical_severity,
        resolution=dataset.resolution,
    )


def generate_dataset_images(
    db: Session,
    dataset: Dataset,
) -> list[GeneratedImage]:

    generated_images = []

    request = build_generation_request(dataset)

    for _ in range(dataset.num_images):

        result = generator.generate(request)

        image_record = GeneratedImage(
            dataset_id=dataset.id,
            image_path=result.image_path,
            generation_model=result.generation_model,
            seed=result.seed,
            prompt=result.prompt,
            status="GENERATED",
        )

        db.add(image_record)
        generated_images.append(image_record)

    db.commit()

    for image in generated_images:
        db.refresh(image)

    return generated_images

def get_dataset_images(
    db: Session,
    dataset_id: int
) -> list[GeneratedImage]:
    return (
        db.query(GeneratedImage)
        .filter(GeneratedImage.dataset_id == dataset_id)
        .order_by(GeneratedImage.id.asc())
        .all()
    )


def get_generated_image_by_id(
    db: Session,
    image_id: int
) -> GeneratedImage | None:
    return (
        db.query(GeneratedImage)
        .filter(GeneratedImage.id == image_id)
        .first()
    )