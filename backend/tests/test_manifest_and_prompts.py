"""
Tests for the generation manifest: prompt building, prompts field, and
JSON serialisation round-trip.

These tests verify:
- build_prompt_from_attributes produces a valid, non-empty string.
- build_manifest always populates the prompts field.
- Prompt length matches requested_count.
- Serialised manifest round-trips correctly (prompts survive JSON encode/decode).
- ManifestProviderConfig and seeds validators work correctly.
- Notebook prompt selection logic (Python re-implementation) rejects None prompts.

All tests are purely local; no Kaggle API or database connections are made.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from generation.manifest import GenerationManifest

import pytest


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _mock_dataset(
    population_group="South Asian",
    region="India",
    age_min=25,
    age_max=35,
    sex_or_attribute="female",
    skin_tone="medium_brown",
    pose="frontal",
    expression="neutral",
    medical_condition=None,
    medical_severity=None,
):
    """Return a simple namespace that mimics the Dataset ORM object."""
    from types import SimpleNamespace
    return SimpleNamespace(
        population_group=population_group,
        region=region,
        age_min=age_min,
        age_max=age_max,
        sex_or_attribute=sex_or_attribute,
        skin_tone=skin_tone,
        pose=pose,
        expression=expression,
        medical_condition=medical_condition,
        medical_severity=medical_severity,
    )


# ---------------------------------------------------------------------------
# Phase B: build_prompt_from_attributes
# ---------------------------------------------------------------------------

class TestBuildPromptFromAttributes:
    """Tests for the canonical prompt-building function."""

    def test_returns_non_empty_string(self):
        from generation.manifest import build_prompt_from_attributes
        prompt = build_prompt_from_attributes(
            population_group="South Asian",
            region="India",
            age_min=25,
            age_max=35,
            sex_or_attribute="female",
            skin_tone="medium_brown",
            pose="frontal",
            expression="neutral",
        )
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 0

    def test_contains_demographic_fields(self):
        from generation.manifest import build_prompt_from_attributes
        prompt = build_prompt_from_attributes(
            population_group="East Asian",
            region="Japan",
            age_min=20,
            age_max=30,
            sex_or_attribute="male",
            skin_tone="fair",
            pose="slight_left",
            expression="smile",
        )
        assert "East Asian" in prompt
        assert "Japan" in prompt
        assert "20" in prompt
        assert "30" in prompt
        assert "male" in prompt
        assert "fair" in prompt
        assert "slight_left" in prompt
        assert "smile" in prompt

    def test_includes_medical_condition_when_present(self):
        from generation.manifest import build_prompt_from_attributes
        prompt = build_prompt_from_attributes(
            population_group="South Asian",
            region="India",
            age_min=40,
            age_max=50,
            sex_or_attribute="female",
            skin_tone="medium_brown",
            pose="frontal",
            expression="neutral",
            medical_condition="rhinitis",
            medical_severity="mild",
        )
        assert "rhinitis" in prompt
        assert "mild" in prompt

    def test_omits_medical_context_when_none(self):
        from generation.manifest import build_prompt_from_attributes
        prompt = build_prompt_from_attributes(
            population_group="South Asian",
            region="India",
            age_min=25,
            age_max=35,
            sex_or_attribute="female",
            skin_tone="medium_brown",
            pose="frontal",
            expression="neutral",
        )
        assert "None" not in prompt
        assert "medical" not in prompt.lower()

    def test_no_secrets_in_prompt(self):
        """Prompt must never contain environment variable names or tokens."""
        from generation.manifest import build_prompt_from_attributes
        prompt = build_prompt_from_attributes(
            population_group="South Asian",
            region="India",
            age_min=25,
            age_max=35,
            sex_or_attribute="female",
            skin_tone="medium_brown",
            pose="frontal",
            expression="neutral",
        )
        forbidden_keywords = [
            "DATABASE_URL", "JWT_SECRET", "KAGGLE_KEY", "KAGGLE_USERNAME",
            "API_KEY", "PASSWORD", "SECRET", "TOKEN",
        ]
        for kw in forbidden_keywords:
            assert kw not in prompt, f"Prompt must not contain '{kw}'"


# ---------------------------------------------------------------------------
# Phase B: prompts field in GenerationManifest
# ---------------------------------------------------------------------------

class TestManifestPromptsField:
    """Tests for the prompts field added to GenerationManifest."""

    def _make_manifest_kwargs(self, **overrides):
        from generation.manifest import ManifestProviderConfig
        base = dict(
            dataset_id=1,
            job_id=1,
            batch_id="test-batch",
            requested_count=1,
            image_width=512,
            image_height=512,
            population_group="South Asian",
            region="India",
            age_min=25,
            age_max=35,
            sex_or_attribute="female",
            skin_tone="medium_brown",
            pose="frontal",
            expression="neutral",
            seeds=[12345],
            provider_config=ManifestProviderConfig(provider="KAGGLE_BATCH"),
        )
        base.update(overrides)
        return base

    def test_prompts_field_defaults_to_none(self):
        """Manifests built without prompts= should default to None (backward compat)."""
        from generation.manifest import GenerationManifest
        m = GenerationManifest(**self._make_manifest_kwargs())
        assert m.prompts is None

    def test_prompts_field_accepts_valid_list(self):
        from generation.manifest import GenerationManifest
        m = GenerationManifest(**self._make_manifest_kwargs(
            prompts=["A fictional portrait of a South Asian woman."]
        ))
        assert m.prompts is not None
        assert len(m.prompts) == 1
        assert "South Asian" in m.prompts[0]

    def test_prompts_length_must_match_requested_count(self):
        from generation.manifest import GenerationManifest
        with pytest.raises(ValueError, match="prompts list length"):
            GenerationManifest(**self._make_manifest_kwargs(
                requested_count=1,
                seeds=[1],
                prompts=["prompt one", "prompt two"],  # length 2 != 1
            ))

    def test_prompts_must_be_non_empty_strings(self):
        from generation.manifest import GenerationManifest
        with pytest.raises(ValueError, match="non-empty"):
            GenerationManifest(**self._make_manifest_kwargs(
                prompts=["   "],  # whitespace-only
            ))

    def test_manifest_serialises_prompts_to_json(self):
        """JSON serialisation includes the prompts list."""
        from generation.manifest import GenerationManifest
        m = GenerationManifest(**self._make_manifest_kwargs(
            prompts=["A synthetic portrait."]
        ))
        js = m.to_json()
        data = json.loads(js)
        assert data["prompts"] == ["A synthetic portrait."]

    def test_manifest_with_none_prompts_serialises_cleanly(self):
        """A manifest without prompts serialises prompts as null."""
        from generation.manifest import GenerationManifest
        m = GenerationManifest(**self._make_manifest_kwargs())
        data = json.loads(m.to_json())
        assert "prompts" in data
        assert data["prompts"] is None

    def test_manifest_round_trips_through_json(self, tmp_path):
        """Save and reload preserves the prompts list."""
        from generation.manifest import GenerationManifest
        m = GenerationManifest(**self._make_manifest_kwargs(
            prompts=["A synthetic portrait test."]
        ))
        path = tmp_path / "manifest.json"
        m.save(path)
        m2 = GenerationManifest.load(path)
        assert m2.prompts == ["A synthetic portrait test."]

    def test_legacy_manifest_without_prompts_loads_as_none(self, tmp_path):
        """Old manifest JSON without a 'prompts' key loads with prompts=None."""
        old_data = {
            "schema_version": "1.0",
            "dataset_id": 1,
            "job_id": 1,
            "batch_id": "old-batch",
            "requested_count": 1,
            "image_width": 512,
            "image_height": 512,
            "population_group": "South Asian",
            "region": "India",
            "age_min": 25,
            "age_max": 35,
            "sex_or_attribute": "female",
            "skin_tone": "medium_brown",
            "pose": "frontal",
            "expression": "neutral",
            "seeds": [99999],
            "provider_config": {
                "provider": "KAGGLE_BATCH",
                "prompt_template_version": "1.0",
                "output_format": "png",
                "num_inference_steps": 12,
                "guidance_scale": 1.0,
                "model_dataset_path": "",
                "base_model_path": "",
            },
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        path = tmp_path / "legacy_manifest.json"
        path.write_text(json.dumps(old_data), encoding="utf-8")
        m = GenerationManifest.load(path)
        assert m.prompts is None


# ---------------------------------------------------------------------------
# Phase B: build_manifest always populates prompts
# ---------------------------------------------------------------------------

class TestBuildManifestPromptsPopulation:
    """build_manifest must always populate the prompts field."""

    def test_build_manifest_populates_prompts_single_image(self):
        from generation.manifest import build_manifest
        ds = _mock_dataset()
        manifest = build_manifest(
            dataset_id=1, job_id=1, dataset=ds,
            requested_count=1, image_width=512, image_height=512,
            provider_name="KAGGLE_BATCH",
        )
        assert manifest.prompts is not None
        assert len(manifest.prompts) == 1
        assert isinstance(manifest.prompts[0], str)
        assert len(manifest.prompts[0].strip()) > 0

    def test_build_manifest_prompts_contains_dataset_attributes(self):
        from generation.manifest import build_manifest
        ds = _mock_dataset(population_group="Southeast Asian", region="Vietnam")
        manifest = build_manifest(
            dataset_id=2, job_id=2, dataset=ds,
            requested_count=1, image_width=512, image_height=512,
            provider_name="KAGGLE_BATCH",
        )
        assert "Southeast Asian" in manifest.prompts[0]
        assert "Vietnam" in manifest.prompts[0]

    def test_build_manifest_prompts_count_matches_requested_count(self):
        from generation.manifest import build_manifest
        ds = _mock_dataset()
        manifest = build_manifest(
            dataset_id=3, job_id=3, dataset=ds,
            requested_count=3, image_width=512, image_height=512,
            provider_name="MOCK",
        )
        assert len(manifest.prompts) == 3

    def test_build_manifest_caller_supplied_prompts_are_used(self):
        """When caller provides prompts=, build_manifest uses them verbatim."""
        from generation.manifest import build_manifest
        ds = _mock_dataset()
        custom = ["Custom prompt for test."]
        manifest = build_manifest(
            dataset_id=4, job_id=4, dataset=ds,
            requested_count=1, image_width=512, image_height=512,
            provider_name="MOCK",
            prompts=custom,
        )
        assert manifest.prompts == custom

    def test_build_manifest_with_medical_context(self):
        from generation.manifest import build_manifest
        ds = _mock_dataset(medical_condition="nasal polyp", medical_severity="moderate")
        manifest = build_manifest(
            dataset_id=5, job_id=5, dataset=ds,
            requested_count=1, image_width=512, image_height=512,
            provider_name="KAGGLE_BATCH",
        )
        assert "nasal polyp" in manifest.prompts[0]
        assert "moderate" in manifest.prompts[0]

    def test_build_manifest_single_image_kaggle(self):
        """Worker calls build_manifest with requested_count=1 for Kaggle."""
        from generation.manifest import build_manifest
        ds = _mock_dataset()
        manifest = build_manifest(
            dataset_id=6, job_id=6, dataset=ds,
            requested_count=1, image_width=512, image_height=512,
            provider_name="KAGGLE_BATCH",
        )
        assert manifest.requested_count == 1
        assert len(manifest.seeds) == 1
        assert len(manifest.prompts) == 1

    def test_build_manifest_json_includes_prompts(self, tmp_path):
        """Serialised manifest JSON contains the prompts array."""
        from generation.manifest import build_manifest
        ds = _mock_dataset()
        manifest = build_manifest(
            dataset_id=7, job_id=7, dataset=ds,
            requested_count=1, image_width=512, image_height=512,
            provider_name="KAGGLE_BATCH",
        )
        path = tmp_path / "manifest.json"
        manifest.save(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "prompts" in raw
        assert isinstance(raw["prompts"], list)
        assert len(raw["prompts"]) == 1


# ---------------------------------------------------------------------------
# Phase G: Notebook prompt logic (Python simulation)
# ---------------------------------------------------------------------------

class TestNotebookPromptLogicSimulation:
    """
    Simulate the notebook manifest-validation cell in plain Python to verify
    the contract without executing the notebook on Kaggle.

    Reflects the patched notebook that REQUIRES the prompts field.
    """

    def _simulate_notebook_validation(self, generation_manifest_dict: dict):
        """Re-implements the patched notebook manifest cell. Returns chosen prompt."""
        GENERATION_MANIFEST = generation_manifest_dict
        GENERATION_SETTINGS = {
            "batch_id": str(GENERATION_MANIFEST.get("batch_id", "batch-001")),
            "requested_count": int(GENERATION_MANIFEST.get("requested_count", 0)),
            "prompts": GENERATION_MANIFEST.get("prompts"),
            "seeds": GENERATION_MANIFEST.get("seeds", []),
            "width": int(GENERATION_MANIFEST.get("width",
                         GENERATION_MANIFEST.get("image_width", 384))),
            "height": int(GENERATION_MANIFEST.get("height",
                          GENERATION_MANIFEST.get("image_height", 384))),
        }

        if GENERATION_SETTINGS["requested_count"] != 1:
            raise ValueError("This notebook is configured to generate exactly one base image.")

        if len(GENERATION_SETTINGS["seeds"]) != GENERATION_SETTINGS["requested_count"]:
            raise ValueError(
                f"Manifest must contain exactly one seed per requested image. "
                f"Requested {GENERATION_SETTINGS['requested_count']}, "
                f"received {len(GENERATION_SETTINGS['seeds'])} seeds."
            )

        prompts_value = GENERATION_SETTINGS["prompts"]
        if prompts_value is None:
            raise ValueError(
                "Manifest is missing the required 'prompts' field. "
                "Ensure the manifest was generated by build_manifest() which always "
                "populates this field from the dataset configuration."
            )
        if not (
            isinstance(prompts_value, list)
            and len(prompts_value) == GENERATION_SETTINGS["requested_count"]
            and all(isinstance(p, str) and p.strip() for p in prompts_value)
        ):
            raise ValueError("Manifest prompts must be a list of non-empty strings, one per image.")

        # Generation loop prompt selection (index=0 for single image)
        index = 0
        prompts = prompts_value
        P01_PROMPT = "HARDCODED_FALLBACK"
        chosen_prompt = prompts[index] if prompts is not None else P01_PROMPT
        return chosen_prompt

    def test_backend_manifest_prompt_reaches_model(self):
        """A manifest from build_manifest passes validation and prompt is used."""
        from generation.manifest import build_manifest
        ds = _mock_dataset()
        m = build_manifest(
            dataset_id=1, job_id=1, dataset=ds,
            requested_count=1, image_width=512, image_height=512,
            provider_name="KAGGLE_BATCH",
        )
        manifest_dict = json.loads(m.to_json())
        chosen_prompt = self._simulate_notebook_validation(manifest_dict)
        assert "South Asian" in chosen_prompt
        assert chosen_prompt != "HARDCODED_FALLBACK"

    def test_manifest_without_prompts_raises_clear_error(self):
        """Missing prompts field raises ValueError, not silent fallback."""
        manifest_dict = {
            "batch_id": "test",
            "requested_count": 1,
            "seeds": [12345],
            "image_width": 512,
            "image_height": 512,
            # No "prompts" key
        }
        with pytest.raises(ValueError, match="missing the required 'prompts' field"):
            self._simulate_notebook_validation(manifest_dict)

    def test_manifest_with_null_prompts_raises_clear_error(self):
        """Explicit null prompts raises ValueError."""
        manifest_dict = {
            "batch_id": "test",
            "requested_count": 1,
            "seeds": [12345],
            "prompts": None,
        }
        with pytest.raises(ValueError, match="missing the required 'prompts' field"):
            self._simulate_notebook_validation(manifest_dict)

    def test_wrong_requested_count_raises(self):
        manifest_dict = {
            "batch_id": "test",
            "requested_count": 2,
            "seeds": [1, 2],
            "prompts": ["p1", "p2"],
        }
        with pytest.raises(ValueError, match="exactly one base image"):
            self._simulate_notebook_validation(manifest_dict)

    def test_seed_count_mismatch_raises(self):
        manifest_dict = {
            "batch_id": "test",
            "requested_count": 1,
            "seeds": [],
            "prompts": ["A portrait."],
        }
        with pytest.raises(ValueError, match="seed"):
            self._simulate_notebook_validation(manifest_dict)

    def test_empty_prompt_string_raises(self):
        manifest_dict = {
            "batch_id": "test",
            "requested_count": 1,
            "seeds": [99],
            "prompts": ["   "],  # whitespace only
        }
        with pytest.raises(ValueError, match="non-empty strings"):
            self._simulate_notebook_validation(manifest_dict)

    def test_backend_prompt_is_used_not_hardcoded(self):
        manifest_dict = {
            "batch_id": "test",
            "requested_count": 1,
            "seeds": [42],
            "prompts": ["Photorealistic portrait of a fictional Southeast Asian adult."],
        }
        chosen = self._simulate_notebook_validation(manifest_dict)
        assert chosen == "Photorealistic portrait of a fictional Southeast Asian adult."
        assert chosen != "HARDCODED_FALLBACK"


# ---------------------------------------------------------------------------
# Phase C: Single-image Kaggle manifest validation
# ---------------------------------------------------------------------------

class TestSingleImageKaggleManifest:
    """Verify the worker produces a valid single-image Kaggle manifest."""

    def test_kaggle_prepare_uses_requested_count_1(self):
        from generation.manifest import build_manifest
        ds = _mock_dataset()
        manifest = build_manifest(
            dataset_id=10, job_id=10, dataset=ds,
            requested_count=1, image_width=512, image_height=512,
            provider_name="KAGGLE_BATCH",
        )
        assert manifest.requested_count == 1
        assert len(manifest.seeds) == 1
        assert manifest.prompts is not None
        assert len(manifest.prompts) == 1

    def test_manifest_provider_config_has_inference_settings(self):
        from generation.manifest import build_manifest
        ds = _mock_dataset()
        manifest = build_manifest(
            dataset_id=11, job_id=11, dataset=ds,
            requested_count=1, image_width=512, image_height=512,
            provider_name="KAGGLE_BATCH",
        )
        assert manifest.provider_config.num_inference_steps == 12
        assert manifest.provider_config.guidance_scale == 1.0
        assert manifest.provider_config.output_format == "png"

    def test_notebook_json_is_valid(self):
        """The fixed notebook file parses as valid JSON with correct structure."""
        nb_path = Path("kaggle_notebook_check/apac-base-image-only.fixed.ipynb")
        if not nb_path.exists():
            pytest.skip("Notebook not present in this environment")
        data = json.loads(nb_path.read_text(encoding="utf-8"))
        assert "cells" in data
        assert data.get("nbformat") == 4
        assert len(data["cells"]) >= 1

    def test_notebook_has_manifest_reading_cell(self):
        """The notebook contains the manifest-reading logic."""
        nb_path = Path("kaggle_notebook_check/apac-base-image-only.fixed.ipynb")
        if not nb_path.exists():
            pytest.skip("Notebook not present in this environment")
        data = json.loads(nb_path.read_text(encoding="utf-8"))
        all_source = " ".join(
            "".join(c.get("source", [])) for c in data["cells"]
        )
        assert "GENERATION_MANIFEST" in all_source
        assert "prompts" in all_source
        assert "batch_summary.json" in all_source
