from sqlalchemy.orm import Session

from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage
from backend.app.schemas.generation import GenerationRequest
from generation.mock_generator import MockFaceGenerator


# Temporary generator.
# Later this will be replaced by Member 2's real
# diffusion-based generator.
generator = MockFaceGenerator()


def build_generation_request(
    dataset: Dataset,
) -> GenerationRequest:
    """
    Convert the stored dataset configuration into the
    standard request format expected by the generation engine.
    """

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
    """
    Generate the requested number of candidate images
    and store their metadata in the database.

    Dataset-level status management is intentionally handled
    by the API route, not by this service.
    """

    generated_images: list[GeneratedImage] = []

    request = build_generation_request(dataset)

    for _ in range(dataset.num_images):

        # Every call to the generator counts as one attempt.
        dataset.generation_attempts += 1

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

        # For the current mock-generation stage,
        # every successfully produced candidate increments
        # generated_count.
        dataset.generated_count += 1

        db.commit()
        db.refresh(image_record)

        generated_images.append(image_record)

    return generated_images


def get_dataset_images(
    db: Session,
    dataset_id: int,
) -> list[GeneratedImage]:
    """
    Retrieve all generated images belonging to a dataset.
    """

    return (
        db.query(GeneratedImage)
        .filter(
            GeneratedImage.dataset_id == dataset_id
        )
        .order_by(
            GeneratedImage.id.asc()
        )
        .all()
    )


def get_generated_image_by_id(
    db: Session,
    image_id: int,
) -> GeneratedImage | None:
    """
    Retrieve a single generated image by its ID.
    """

    return (
        db.query(GeneratedImage)
        .filter(
            GeneratedImage.id == image_id
        )
        .first()
    )