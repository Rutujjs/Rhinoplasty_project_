from pathlib import Path
from random import randint
from time import perf_counter

from PIL import Image, ImageDraw

from backend.app.schemas.generation import (
    GenerationRequest,
    GeneratedImageResult,
)
from generation.interface import FaceGenerator


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "storage"
    / "generated"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


class MockFaceGenerator(FaceGenerator):

    def generate(
        self,
        request: GenerationRequest,
    ) -> GeneratedImageResult:

        start_time = perf_counter()

        seed = randint(
            100000,
            999999
        )

        prompt = (
            f"Photorealistic synthetic portrait of a "
            f"fictional person, "
            f"population group {request.population_group}, "
            f"age {request.age_min} to {request.age_max}, "
            f"sex/attribute {request.sex_or_attribute}, "
            f"skin tone {request.skin_tone}, "
            f"pose {request.pose}, "
            f"expression {request.expression}, "
            f"region {request.region}."
        )

        image = Image.new(
            "RGB",
            (
                request.resolution,
                request.resolution
            ),
            "white",
        )

        draw = ImageDraw.Draw(image)

        draw.text(
            (30, 30),
            "MOCK SYNTHETIC FACE",
            fill="black",
        )

        draw.text(
            (30, 60),
            f"Seed: {seed}",
            fill="black",
        )

        filename = f"mock_{seed}.png"

        file_path = (
            OUTPUT_DIR / filename
        )

        image.save(file_path)

        generation_time = (
            perf_counter() - start_time
        )

        return GeneratedImageResult(
            image_path=str(file_path),
            generation_model="MOCK",
            seed=seed,
            prompt=prompt,
            generation_time=generation_time,
        )