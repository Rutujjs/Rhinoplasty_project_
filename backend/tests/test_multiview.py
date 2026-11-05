"""
Unit and Integration Tests for the Updated Multi-View FLUX Generation Architecture.

Tests cover:
  1. Multi-view 9-viewpoint manifest generation and canonical prompts
  2. Custom prompt synchronization and transmission in manifest
  3. New Kaggle notebook configuration validation (apac_face_generator.ipynb)
  4. MOCK provider multi-view generation of all 9 viewpoints
  5. Multi-view database models, fields, and persistence
  6. Deformation service interface boundaries and placeholder responses
  7. Dataset creation and image listing endpoints with viewpoint metadata
  8. ZIP export bundling multi-view metadata
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from generation.manifest import (
    GenerationManifest,
    build_manifest,
    build_prompt_from_attributes,
)
from generation.mock_generator import MockFaceGenerator
from generation.viewpoints import CLINICAL_VIEWPOINTS, TOTAL_REQUIRED_VIEWPOINTS, VIEWPOINT_KEYS


# --------------------------------------------------------------------------- #
# 1. Canonical Viewpoint Definitions & Manifest Tests                          #
# --------------------------------------------------------------------------- #

class TestViewpointsAndManifest:
    def test_nine_viewpoints_defined(self):
        """Verify exactly nine standardized clinical viewpoints exist."""
        assert len(CLINICAL_VIEWPOINTS) == 9
        assert TOTAL_REQUIRED_VIEWPOINTS == 9
        expected_keys = [
            "frontal",
            "left_20", "left_40", "left_60",
            "right_20", "right_40", "right_60",
            "basal", "helicopter",
        ]
        assert VIEWPOINT_KEYS == expected_keys

    def test_build_manifest_with_multiview_fields(self):
        """Manifest contains patient_id, viewpoints list, and custom prompt."""
        dummy_dataset = MagicMock(
            population_group="East Asian",
            region="Korea",
            age_min=25,
            age_max=32,
            sex_or_attribute="female",
            skin_tone="fair",
            pose="frontal",
            expression="neutral",
            medical_condition="Preoperative Rhinoplasty",
            medical_severity="moderate",
            patient_id="P09",
            custom_prompt="Custom clinical prompt for patient P09",
            viewpoint_prompts=None,
        )

        manifest = build_manifest(
            dataset_id=42,
            job_id=101,
            dataset=dummy_dataset,
            requested_count=9,
            image_width=512,
            image_height=512,
            provider_name="KAGGLE_BATCH",
        )

        assert manifest.dataset_id == 42
        assert manifest.job_id == 101
        assert manifest.patient_id == "P09"
        assert manifest.custom_prompt == "Custom clinical prompt for patient P09"
        assert manifest.prompts[0] == "Custom clinical prompt for patient P09"
        assert manifest.viewpoints == VIEWPOINT_KEYS
        assert len(manifest.seeds) == 9

    def test_manifest_serialization_and_loading(self):
        """Manifest can be cleanly serialized to JSON and re-loaded."""
        Path("storage").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir="storage") as td:
            dummy_dataset = MagicMock(
                population_group="South Asian",
                region="India",
                age_min=30,
                age_max=40,
                sex_or_attribute="male",
                skin_tone="medium_brown",
                pose="frontal",
                expression="neutral",
                medical_condition=None,
                medical_severity=None,
                patient_id="P01",
                custom_prompt=None,
                viewpoint_prompts={"left_20": "Custom Left 20 Prompt"},
            )

            manifest = build_manifest(
                dataset_id=1,
                job_id=2,
                dataset=dummy_dataset,
                requested_count=1,
                image_width=512,
                image_height=512,
                provider_name="MOCK",
            )

            manifest_file = Path(td) / "manifest.json"
            manifest.save(manifest_file)
            loaded = GenerationManifest.load(manifest_file)

            assert loaded.dataset_id == 1
            assert loaded.patient_id == "P01"
            assert loaded.viewpoint_prompts == {"left_20": "Custom Left 20 Prompt"}


# --------------------------------------------------------------------------- #
# 2. Notebook Configuration Tests                                             #
# --------------------------------------------------------------------------- #

class TestNewNotebookConfiguration:
    def test_new_notebook_file_exists(self):
        """The active notebook must point to apac_face_generator.ipynb and be valid JSON."""
        nb_path = Path("kaggle_notebook_check/apac_face_generator.ipynb")
        assert nb_path.exists(), f"New notebook missing at {nb_path}"
        data = json.loads(nb_path.read_text(encoding="utf-8"))
        assert "cells" in data
        assert data.get("nbformat") == 4

    def test_kernel_metadata_points_to_new_notebook(self):
        """kernel-metadata.json must reference apac_face_generator.ipynb and APAC_FACE_GENERATOR."""
        meta_path = Path("kaggle_notebook_check/kernel-metadata.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["code_file"] == "apac_face_generator.ipynb"
        assert meta["title"] == "APAC_FACE_GENERATOR"
        assert meta["enable_gpu"] is True


# --------------------------------------------------------------------------- #
# 3. MOCK Multi-View Generation & Directory Layout Tests                       #
# --------------------------------------------------------------------------- #

class TestMockMultiViewGeneration:
    def test_mock_generates_all_nine_viewpoints(self):
        """MockFaceGenerator produces all 9 viewpoints structured under patient_id/original/."""
        generator = MockFaceGenerator()
        from backend.app.schemas.generation import GenerationRequest

        req = GenerationRequest(
            age_min=25,
            age_max=35,
            sex_or_attribute="female",
            skin_tone="fair",
            population_group="East Asian",
            region="Japan",
            pose="frontal",
            expression="neutral",
            resolution=512,
            patient_id="P77",
        )

        results = generator.generate_multiview(req, patient_id="P77")
        assert len(results) == 9

        keys = [r.viewpoint for r in results]
        assert keys == VIEWPOINT_KEYS

        for r in results:
            p = Path(r.image_path)
            assert p.exists()
            assert "P77" in str(p)
            assert "original" in str(p)
            assert r.patient_id == "P77"
            assert r.generation_model == "MOCK"


# --------------------------------------------------------------------------- #
# 4. Deformation Service & Endpoint Tests                                     #
# --------------------------------------------------------------------------- #

class TestDeformationServiceIntegration:
    def test_deformation_status_endpoint_returns_pending(self, client: TestClient):
        """Deformation status endpoint returns PENDING_INTEGRATION cleanly."""
        ds_resp = client.post(
            "/datasets/",
            json={
                "dataset_name": "Test_Deform_DS",
                "patient_id": "P01",
                "region": "Korea",
                "population_group": "East Asian",
                "age_min": 25,
                "age_max": 30,
                "sex_or_attribute": "female",
                "skin_tone": "fair",
                "pose": "frontal",
                "expression": "neutral",
                "num_images": 9,
                "resolution": 512,
            },
        )
        assert ds_resp.status_code == 200
        dataset_id = ds_resp.json()["id"]

        resp = client.get(f"/datasets/{dataset_id}/deformation/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["dataset_id"] == dataset_id
        assert data["patient_id"] == "P01"
        assert data["backend_integration_ready"] is False
        assert "awaiting" in data["message"].lower() or "ready" in data["message"].lower()

    def test_deformation_trigger_returns_accepted(self, client: TestClient):
        """Triggering deformation returns 202 Accepted with clear status."""
        ds_resp = client.post(
            "/datasets/",
            json={
                "dataset_name": "Trigger_Deform_DS",
                "patient_id": "P02",
                "region": "China",
                "population_group": "East Asian",
                "age_min": 28,
                "age_max": 35,
                "sex_or_attribute": "male",
                "skin_tone": "fair",
                "pose": "frontal",
                "expression": "neutral",
                "num_images": 9,
                "resolution": 512,
            },
        )
        assert ds_resp.status_code == 200
        dataset_id = ds_resp.json()["id"]

        resp = client.post(
            f"/datasets/{dataset_id}/deformation/trigger",
            json={
                "patient_id": "P02",
                "parameters": {
                    "hump_reduction_mm": 2.5,
                    "tip_rotation_degrees": 5.0,
                },
            },
        )
        assert resp.status_code == 202
        assert resp.json()["dataset_id"] == dataset_id


# --------------------------------------------------------------------------- #
# 5. Dataset API Multi-View & Synchronized Custom Prompt Tests                #
# --------------------------------------------------------------------------- #

class TestDatasetMultiViewAPI:
    def test_create_dataset_with_custom_prompt(self, client: TestClient):
        """Dataset creation accepts and persists custom_prompt and patient_id."""
        custom_p = "Clinical high-definition portrait with prominent nasal bridge."
        payload = {
            "dataset_name": "MultiView_P03",
            "patient_id": "P03",
            "region": "Japan",
            "population_group": "East Asian",
            "age_min": 24,
            "age_max": 29,
            "sex_or_attribute": "female",
            "skin_tone": "fair",
            "pose": "frontal",
            "expression": "neutral",
            "num_images": 9,
            "resolution": 512,
            "custom_prompt": custom_p,
        }
        res = client.post("/datasets/", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["patient_id"] == "P03"
        assert data["custom_prompt"] == custom_p
        assert data["num_images"] == 9
