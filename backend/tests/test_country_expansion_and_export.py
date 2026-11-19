"""
Tests for Update 1 (Dataset Export Fix) and Update 2 (Country Expansion & Languages.pdf).
"""

import io
import json
import zipfile
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.models.dataset import Dataset
from backend.app.services.export_service import build_export_zip
from backend.app.services.language_service import (
    LANGUAGE_STRATEGY,
    ROLLOUT_PHASES,
    STANDARDIZED_COUNTRIES,
    language_service,
)
from backend.tests.conftest import create_test_dataset
from generation.manifest import build_prompt_from_attributes


def _run_worker_once(db_session: Session):
    """Run one mock worker cycle using the test DB session."""
    from backend.app.services.job_service import get_queued_jobs, claim_job
    from backend.worker import _process_job, WORKER_ID

    jobs = get_queued_jobs(db_session, limit=10)
    for job in jobs:
        claim_job(db_session, job, worker_id=WORKER_ID)
        _process_job(db_session, job)


# =========================================================================== #
# UPDATE 1: DATASET EXPORT TESTS                                              #
# =========================================================================== #

class TestDatasetExport:
    def test_get_export_completed_dataset_success(self, client: TestClient, db_session: Session):
        """GET /datasets/{id}/export builds and downloads comprehensive ZIP archive."""
        dataset = create_test_dataset(client, overrides={"num_images": 2, "patient_id": "P01"})
        ds_id = dataset["id"]

        # Enqueue and process generation
        client.post(f"/datasets/{ds_id}/generate")
        _run_worker_once(db_session)

        # Request export via GET
        resp = client.get(f"/datasets/{ds_id}/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert f'filename="dataset_{ds_id}_export.zip"' in resp.headers["content-disposition"]

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = zf.namelist()

        # Check required files per user specification
        assert "dataset_metadata.json" in names
        assert "manifest.txt" in names
        assert "README.txt" in names
        assert "deformation_audit.json" in names
        assert "patients/P01/metadata.json" in names

        # Check metadata contents
        meta = json.loads(zf.read("dataset_metadata.json"))
        assert meta["dataset"]["id"] == ds_id
        assert meta["dataset"]["patient_id"] == "P01"
        assert meta["export"]["total_images"] >= 2

    def test_post_export_completed_dataset_success(self, client: TestClient, db_session: Session):
        """POST /datasets/{id}/export also builds and downloads the ZIP archive."""
        dataset = create_test_dataset(client, overrides={"num_images": 1, "patient_id": "P01"})
        ds_id = dataset["id"]

        client.post(f"/datasets/{ds_id}/generate")
        _run_worker_once(db_session)

        resp = client.post(f"/datasets/{ds_id}/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert f'filename="dataset_{ds_id}_export.zip"' in resp.headers["content-disposition"]

    def test_export_nonexistent_dataset_returns_404(self, client: TestClient):
        """GET and POST for nonexistent dataset return 404."""
        resp_get = client.get("/datasets/999999/export")
        assert resp_get.status_code == 404

        resp_post = client.post("/datasets/999999/export")
        assert resp_post.status_code == 404

    def test_export_uncompleted_dataset_without_zip_returns_404(self, client: TestClient):
        """GET for uncompleted dataset with no pre-existing ZIP returns 404."""
        dataset = create_test_dataset(client)
        resp = client.get(f"/datasets/{dataset['id']}/export")
        assert resp.status_code == 404


# =========================================================================== #
# UPDATE 2: COUNTRY & LANGUAGE EXPANSION TESTS                                #
# =========================================================================== #

class TestCountryLanguageExpansion:
    def test_languages_catalog(self, client: TestClient):
        """GET /languages returns all languages from Languages.pdf with priority and phases."""
        resp = client.get("/languages")
        assert resp.status_code == 200
        data = resp.json()

        lang_map = {l["language"]: l for l in data}

        # Check Phase 1 High Priority
        phase1_expected = ["Spanish", "Arabic", "German", "French", "Turkish"]
        for lang in phase1_expected:
            assert lang in lang_map
            assert lang_map[lang]["priority"] == "High"
            assert lang_map[lang]["phase"] == "Phase 1"

        # Check Phase 2 Medium Priority
        phase2_expected = ["Italian", "Mandarin Chinese", "Dutch", "Malay", "Tamil"]
        for lang in phase2_expected:
            assert lang in lang_map
            assert lang_map[lang]["priority"] == "Medium"
            assert lang_map[lang]["phase"] == "Phase 2"

        # Check Phase 3 Selective Priority
        phase3_expected = ["Hindi", "Indonesian", "Russian", "Malayalam", "Telugu"]
        for lang in phase3_expected:
            assert lang in lang_map
            assert lang_map[lang]["priority"] == "Selective"
            assert lang_map[lang]["phase"] == "Phase 3"

        # Check Currently Supported Languages preserved
        supported_expected = ["English", "Japanese", "Portuguese", "Thai", "Korean"]
        for lang in supported_expected:
            assert lang in lang_map
            assert lang_map[lang]["priority"] == "Supported"

        # Market categories validation: Russian and Malayalam preserve market categories
        assert "Russian-speaking markets" in lang_map["Russian"]["market_categories"]
        assert "Overseas communities" in lang_map["Malayalam"]["market_categories"]

    def test_countries_catalog(self, client: TestClient):
        """GET /languages/countries returns all unique countries and many-to-many language mappings."""
        resp = client.get("/languages/countries")
        assert resp.status_code == 200
        countries = {c["country"]: c for c in resp.json()}

        # Check unique target countries from Languages.pdf
        expected_countries = [
            "Spain", "Mexico", "Colombia",
            "UAE", "Saudi Arabia", "Qatar",
            "Germany", "Austria", "Switzerland",
            "France", "Belgium", "Turkey",
            "Italy", "China", "Taiwan", "Netherlands",
            "Malaysia", "Brunei", "Singapore",
            "India", "Sri Lanka", "Indonesia", "Russia",
            "Japan", "Korea", "Thailand", "Vietnam",
            "Brazil", "Portugal", "United States", "United Kingdom", "Australia",
        ]
        for country in expected_countries:
            assert country in countries, f"Missing country: {country}"
            assert len(countries[country]["iso2"]) == 2
            assert len(countries[country]["iso3"]) == 3

        # Multi-language verification from Languages.pdf
        ch = countries["Switzerland"]["associated_languages"]
        assert "German" in ch and "French" in ch and "Italian" in ch

        be = countries["Belgium"]["associated_languages"]
        assert "French" in be and "Dutch" in be

        ind = countries["India"]["associated_languages"]
        assert "Hindi" in ind and "Tamil" in ind and "Malayalam" in ind and "Telugu" in ind

        sg = countries["Singapore"]["associated_languages"]
        assert "Tamil" in sg and "English" in sg

        # Verify market categories are NOT countries
        assert "Russian-speaking markets" not in countries
        assert "Overseas communities" not in countries

    def test_rollout_phases(self, client: TestClient):
        """GET /languages/phases returns 3 distinct rollout phases matching Languages.pdf."""
        resp = client.get("/languages/phases")
        assert resp.status_code == 200
        phases = resp.json()
        assert len(phases) == 3

        p1 = phases[0]
        assert p1["phase"] == "Phase 1"
        assert p1["priority"] == "High"
        assert "Spanish" in p1["languages"]
        assert "Arabic" in p1["languages"]
        assert "Target the most promising international aesthetic markets" in p1["objective"]

        p2 = phases[1]
        assert p2["phase"] == "Phase 2"
        assert "Expand further and support Singapore" in p2["objective"]

        p3 = phases[2]
        assert p3["phase"] == "Phase 3"
        assert "Expand selectively, based on measured demand" in p3["objective"]

    def test_download_strategy_pdf(self, client: TestClient):
        """GET /languages/report/pdf generates and serves the Language Expansion Strategy PDF."""
        import fitz
        resp = client.get("/languages/report/pdf")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"
        assert "Faceify_Labs_Language_Expansion_Strategy.pdf" in resp.headers["content-disposition"]

        doc = fitz.open(stream=resp.content, filetype="pdf")
        assert len(doc) >= 1
        full_text = "".join(page.get_text() for page in doc)

        assert "FACEIFY LABS" in full_text
        assert "Language Expansion Strategy" in full_text
        assert "Currently supported: English, Japanese, Portuguese, Thai, Korean." in full_text
        assert "Spanish" in full_text
        assert "Spain, Mexico, Colombia" in full_text
        assert "UAE, Saudi Arabia, Qatar" in full_text
        assert "Rollout phases" in full_text

    def test_prompt_generation_with_new_countries(self):
        """Verify prompt generation works seamlessly with newly added target countries."""
        prompt_spain = build_prompt_from_attributes(
            population_group="European",
            region="Spain",
            age_min=28,
            age_max=35,
            sex_or_attribute="female",
            skin_tone="fair",
            pose="frontal",
            expression="neutral relaxed",
            medical_condition="Preoperative rhinoplasty nasal dorsum assessment",
            medical_severity="moderate",
        )
        assert "European, Spain region" in prompt_spain
        assert "Spain" in prompt_spain

        prompt_uae = build_prompt_from_attributes(
            population_group="Middle Eastern",
            region="UAE",
            age_min=25,
            age_max=32,
            sex_or_attribute="female",
            skin_tone="olive",
            pose="frontal",
            expression="neutral relaxed",
            medical_condition="Preoperative rhinoplasty nasal dorsum assessment",
            medical_severity="moderate",
        )
        assert "Middle Eastern, UAE region" in prompt_uae
        assert "UAE" in prompt_uae
