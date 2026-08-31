from pathlib import Path
from random import randint
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "storage" / "generated"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def generate_mock_image(
    prompt: str,
    resolution: int = 512,
) -> dict:

    seed = randint(100000, 999999)

    image = Image.new(
        "RGB",
        (resolution, resolution),
        "white"
    )

    draw = ImageDraw.Draw(image)

    draw.text(
        (30, 30),
        "MOCK SYNTHETIC FACE",
        fill="black"
    )

    draw.text(
        (30, 60),
        f"Seed: {seed}",
        fill="black"
    )

    file_name = f"mock_{seed}.png"
    file_path = OUTPUT_DIR / file_name

    image.save(file_path)

    return {
        "image_path": str(file_path),
        "generation_model": "MOCK",
        "seed": seed,
        "prompt": prompt,
    }