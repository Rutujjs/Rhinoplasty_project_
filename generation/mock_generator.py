"""
MockFaceGenerator – placeholder image generator for development and tests.

Supports both single-image legacy generation and full multi-view clinical generation
(9 viewpoints: Frontal 0°, Left 20°, Left 40°, Left 60°, Right 20°, Right 40°, Right 60°,
Basal, Helicopter) for testing and development without a GPU or Kaggle account.
"""

from pathlib import Path
from random import randint
from time import perf_counter
from typing import List, Optional

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
        """Load and resize a photorealistic reference identity image for this viewpoint."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        res = request.resolution or 512

        # Viewpoint filename search patterns
        pats = {
            "frontal": ["*frontal*", "*FRONTAL*"],
            "left_20": ["*left_20*", "*LEFT20*"],
            "left_40": ["*left_40*", "*LEFT40*"],
            "left_60": ["*left_60*", "*LEFT60*"],
            "right_20": ["*right_20*", "*RIGHT20*"],
            "right_40": ["*right_40*", "*RIGHT40*"],
            "right_60": ["*right_60*", "*RIGHT60*"],
            "basal": ["*basal*", "*BASAL*"],
            "helicopter": ["*helicopter*", "*HELICOPTER*"],
        }
        target_pats = pats.get(viewpoint_spec.key, ["*frontal*"])

        # Available high-quality photorealistic synthetic identities
        available_ids = ["H01", "H02", "H03", "H06", "H07", "H08", "H09"]
        selected_id = "H01"
        if patient_id and patient_id.upper() in available_ids:
            selected_id = patient_id.upper()
        else:
            # Deterministically choose an identity based on patient_id and seed
            hash_val = sum(ord(c) for c in (patient_id or "P01")) + (seed or 0)
            selected_id = available_ids[hash_val % len(available_ids)]

        ident_dir = PROJECT_ROOT / "nasal-deformation-pipeline-main" / "data" / "identities" / selected_id / "input"
        matched_file: Path | None = None
        if ident_dir.exists():
            for pat in target_pats:
                matches = list(ident_dir.glob(f"{pat}.png"))
                if matches:
                    matched_file = matches[0]
                    break

        if not matched_file:
            # Fallback to test gallery P01
            gallery_file = PROJECT_ROOT / "frontend" / "test_gallery" / "P01" / f"{viewpoint_spec.key}.png"
            if gallery_file.exists():
                matched_file = gallery_file

        if matched_file and matched_file.exists():
            with Image.open(matched_file) as src_img:
                src_img = src_img.convert("RGB")
                if src_img.size != (res, res):
                    src_img = src_img.resize((res, res), Image.Resampling.LANCZOS)
                src_img.save(dest_path, "PNG")
        else:
            # Extreme fallback: create a clean neutral medical portrait background
            image = Image.new("RGB", (res, res), (235, 238, 242))
            draw = ImageDraw.Draw(image)
            cx, cy = res // 2, res // 2
            r = min(res // 3, 140)
            draw.ellipse([(cx - r, cy - int(r * 1.3)), (cx + r, cy + int(r * 1.3))], fill=(210, 190, 175))
            image.save(dest_path, "PNG")

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
        dataset_id: Optional[int] = None,
    ) -> List[GeneratedImageResult]:
        """Generate all nine clinical viewpoints for one synthetic patient."""
        results: List[GeneratedImageResult] = []
        base_seed = randint(100_000, 800_000)

        # Prepare dataset-isolated directory if dataset_id provided, otherwise fallback to patient_root
        if dataset_id is not None:
            patient_root = PROJECT_ROOT / "storage" / "datasets" / f"dataset_{dataset_id}" / patient_id
        else:
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
