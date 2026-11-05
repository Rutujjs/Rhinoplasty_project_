"""
MockFaceGenerator – placeholder image generator for development and tests.

Supports both single-image legacy generation and full multi-view clinical generation
(9 viewpoints: Frontal 0°, Left 20°, Left 40°, Left 60°, Right 20°, Right 40°, Right 60°,
Basal, Helicopter) for testing and development without a GPU or Kaggle account.
"""

from pathlib import Path
from random import randint
from time import perf_counter
from typing import List

from PIL import Image, ImageDraw

from backend.app.schemas.generation import (
    GenerationRequest,
    GeneratedImageResult,
)
from generation.interface import FaceGenerator
from generation.viewpoints import CLINICAL_VIEWPOINTS, ViewpointSpec


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "storage" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MockFaceGenerator(FaceGenerator):
    """
    Synchronous mock generator.

    Creates structured PNG images annotated with viewpoint and demographic parameters
    so the full multi-view pipeline (database records, viewpoint galleries, storage
    organization, validation) can be verified offline.
    """

    @property
    def provider_name(self) -> str:
        return "MOCK"

    def _draw_mock_image(
        self,
        request: GenerationRequest,
        seed: int,
        viewpoint_spec: ViewpointSpec,
        patient_id: str,
        dest_path: Path,
    ) -> None:
        """Helper to draw a clear clinical placeholder image with viewpoint metadata."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        res = request.resolution or 512
        # Different subtle backgrounds to easily tell viewpoints apart in the gallery
        bg_colors = {
            "frontal": (245, 247, 250),
            "left_20": (240, 245, 255),
            "left_40": (235, 240, 255),
            "left_60": (230, 235, 250),
            "right_20": (255, 245, 240),
            "right_40": (255, 240, 235),
            "right_60": (250, 235, 230),
            "basal": (240, 255, 245),
            "helicopter": (255, 255, 240),
        }
        bg = bg_colors.get(viewpoint_spec.key, (245, 247, 250))

        image = Image.new("RGB", (res, res), bg)
        draw = ImageDraw.Draw(image)

        # Draw a framing border
        draw.rectangle([(8, 8), (res - 8, res - 8)], outline=(180, 190, 205), width=2)

        # Draw title and viewpoint
        draw.text((25, 25), "SYNTHETIC CLINICAL MULTI-VIEW", fill=(40, 60, 90))
        draw.text((25, 55), f"VIEWPOINT: {viewpoint_spec.name.upper()}", fill=(20, 30, 60))
        draw.text((25, 80), f"Orientation: {viewpoint_spec.nose_toward}", fill=(100, 110, 130))
        draw.text((25, 105), f"Patient ID: {patient_id}", fill=(30, 90, 160))
        draw.text((25, 130), f"Seed: {seed}", fill=(70, 70, 80))

        # Demographic labels
        draw.text((25, 165), f"Group: {request.population_group} ({request.region})", fill=(90, 95, 105))
        draw.text((25, 190), f"Age: {request.age_min}–{request.age_max} | Sex: {request.sex_or_attribute}", fill=(90, 95, 105))
        draw.text((25, 215), f"Skin Tone: {request.skin_tone}", fill=(90, 95, 105))
        if request.medical_condition:
            draw.text((25, 240), f"Condition: {request.medical_condition} ({request.medical_severity or 'std'})", fill=(180, 60, 50))

        # Draw a visual representation of the camera/head angle
        cx, cy = res // 2, res // 2 + 60
        r = min(res // 5, 70)
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=(120, 140, 170), width=2)
        # Nose indicator direction
        if viewpoint_spec.key == "frontal":
            draw.polygon([(cx, cy - 10), (cx - 8, cy + 15), (cx + 8, cy + 15)], fill=(70, 130, 180))
        elif "left" in viewpoint_spec.key:
            draw.polygon([(cx - 20, cy), (cx, cy - 12), (cx, cy + 12)], fill=(60, 120, 210))
        elif "right" in viewpoint_spec.key:
            draw.polygon([(cx + 20, cy), (cx, cy - 12), (cx, cy + 12)], fill=(210, 90, 50))
        elif viewpoint_spec.key == "basal":
            draw.polygon([(cx, cy + 20), (cx - 15, cy), (cx + 15, cy)], fill=(40, 160, 90))
        elif viewpoint_spec.key == "helicopter":
            draw.polygon([(cx, cy - 20), (cx - 15, cy), (cx + 15, cy)], fill=(180, 160, 40))

        image.save(dest_path)

    def generate(
        self,
        request: GenerationRequest,
    ) -> GeneratedImageResult:
        """Single-image generation (backward compatible with Phase 1 API contracts)."""
        start_time = perf_counter()
        seed = randint(100_000, 999_999)

        from generation.manifest import build_prompt_from_attributes
        prompt = request.custom_prompt or build_prompt_from_attributes(
            population_group=request.population_group,
            region=request.region,
            age_min=request.age_min,
            age_max=request.age_max,
            sex_or_attribute=request.sex_or_attribute,
            skin_tone=request.skin_tone,
            pose=request.pose,
            expression=request.expression,
            medical_condition=request.medical_condition,
            medical_severity=request.medical_severity,
        )

        patient_id = request.patient_id or "P01"
        view_key = request.viewpoint or "frontal"
        spec = next((v for v in CLINICAL_VIEWPOINTS if v.key == view_key), CLINICAL_VIEWPOINTS[0])

        # New multi-view path structure
        file_path = OUTPUT_DIR / patient_id / "original" / f"{view_key}.png"
        self._draw_mock_image(request, seed, spec, patient_id, file_path)

        # Also write root mock_{seed}.png for legacy test cases expecting flat files
        legacy_path = OUTPUT_DIR / f"mock_{seed}.png"
        legacy_path.write_bytes(file_path.read_bytes())

        generation_time = perf_counter() - start_time
        return GeneratedImageResult(
            image_path=str(file_path),
            generation_model="MOCK",
            seed=seed,
            prompt=prompt,
            generation_time=generation_time,
            patient_id=patient_id,
            viewpoint=view_key,
            viewpoint_name=spec.name,
        )

    def generate_multiview(
        self,
        request: GenerationRequest,
        patient_id: str = "P01",
    ) -> List[GeneratedImageResult]:
        """Generate all nine clinical viewpoints for one synthetic patient."""
        results: List[GeneratedImageResult] = []
        base_seed = randint(100_000, 800_000)

        # Prepare future integration directories
        patient_root = OUTPUT_DIR / patient_id
        (patient_root / "original").mkdir(parents=True, exist_ok=True)
        (patient_root / "deformed").mkdir(parents=True, exist_ok=True)
        (patient_root / "masks").mkdir(parents=True, exist_ok=True)

        from generation.manifest import build_prompt_from_attributes
        base_prompt = request.custom_prompt or build_prompt_from_attributes(
            population_group=request.population_group,
            region=request.region,
            age_min=request.age_min,
            age_max=request.age_max,
            sex_or_attribute=request.sex_or_attribute,
            skin_tone=request.skin_tone,
            pose=request.pose,
            expression=request.expression,
            medical_condition=request.medical_condition,
            medical_severity=request.medical_severity,
        )

        for idx, spec in enumerate(CLINICAL_VIEWPOINTS):
            start_time = perf_counter()
            seed = base_seed + (idx * 100)
            dest_file = patient_root / "original" / f"{spec.key}.png"

            self._draw_mock_image(request, seed, spec, patient_id, dest_file)
            gen_time = perf_counter() - start_time

            results.append(
                GeneratedImageResult(
                    image_path=str(dest_file),
                    generation_model="MOCK",
                    seed=seed,
                    prompt=f"{spec.name}: {base_prompt}",
                    generation_time=gen_time,
                    patient_id=patient_id,
                    viewpoint=spec.key,
                    viewpoint_name=spec.name,
                )
            )

        return results
