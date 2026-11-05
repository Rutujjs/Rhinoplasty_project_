"""
Tests for ZIP export endpoints.

POST /datasets/{id}/export  – build and download ZIP
GET  /datasets/{id}/export  – download existing ZIP

These tests run the MOCK worker inline to produce real images, then
exercise the export endpoints.
"""
import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import create_test_dataset


def _run_worker_once(db_session):
    """Run one worker cycle using the test DB session."""
    from backend.app.services.job_service import get_queued_jobs, claim_job
    from backend.worker import _process_job, WORKER_ID

    jobs = get_queued_jobs(db_session, limit=10)
    for job in jobs:
        claim_job(db_session, job, worker_id=WORKER_ID)
        _process_job(db_session, job)


class TestExportEndpoints:
    def test_export_nonexistent_dataset_returns_404(self, client: TestClient):
        response = client.post("/datasets/999999/export")
        assert response.status_code == 404

    def test_export_uncompleted_dataset_returns_500(self, client: TestClient):
        """Export is only allowed for COMPLETED datasets."""
        dataset = create_test_dataset(client)
        response = client.post(f"/datasets/{dataset['id']}/export")
        assert response.status_code == 500

    def test_get_export_nonexistent_returns_404(self, client: TestClient, tmp_path):
        """GET /export with no existing ZIP returns 404."""
        from unittest.mock import patch
        dataset = create_test_dataset(client)
        # Override EXPORT_DIR to a guaranteed-empty temp dir.
        with patch("backend.app.api.routes.datasets.settings") as mock_settings:
            mock_settings.EXPORT_DIR = str(tmp_path)
            response = client.get(f"/datasets/{dataset['id']}/export")
        assert response.status_code == 404

    def test_export_completed_dataset_returns_zip(
        self, client: TestClient, db_session
    ):
        """Full flow: create → enqueue → run worker → export."""
        dataset = create_test_dataset(client, overrides={"num_images": 2})
        dataset_id = dataset["id"]

        # Enqueue generation job.
        r = client.post(f"/datasets/{dataset_id}/generate")
        assert r.status_code == 202

        # Run the mock worker inline.
        _run_worker_once(db_session)

        # Verify dataset is now COMPLETED.
        status = client.get(f"/datasets/{dataset_id}/generation-status")
        assert status.json()["status"] == "COMPLETED"

        # Request export.
        export_response = client.post(f"/datasets/{dataset_id}/export")
        assert export_response.status_code == 200
        assert export_response.headers["content-type"] == "application/zip"

        # Validate the ZIP contents.
        zf = zipfile.ZipFile(io.BytesIO(export_response.content))
        names = zf.namelist()
        assert any("metadata.json" in n for n in names)
        assert any("manifest.txt" in n for n in names)

    def test_export_zip_contains_metadata(
        self, client: TestClient, db_session
    ):
        dataset = create_test_dataset(client, overrides={"num_images": 1})
        dataset_id = dataset["id"]
        client.post(f"/datasets/{dataset_id}/generate")
        _run_worker_once(db_session)

        export_response = client.post(f"/datasets/{dataset_id}/export")
        assert export_response.status_code == 200

        zf = zipfile.ZipFile(io.BytesIO(export_response.content))
        import json
        metadata_name = next(n for n in zf.namelist() if "metadata.json" in n)
        metadata = json.loads(zf.read(metadata_name))

        assert metadata["dataset"]["id"] == dataset_id
        assert metadata["export"]["total_images"] >= 1
        assert len(metadata["images"]) >= 1

    def test_get_export_after_post_returns_zip(
        self, client: TestClient, db_session
    ):
        """GET /export returns the same file as POST /export after it's built."""
        dataset = create_test_dataset(client, overrides={"num_images": 1})
        dataset_id = dataset["id"]
        client.post(f"/datasets/{dataset_id}/generate")
        _run_worker_once(db_session)

        # Build it first.
        post_resp = client.post(f"/datasets/{dataset_id}/export")
        assert post_resp.status_code == 200

        # GET should now return the file.
        get_resp = client.get(f"/datasets/{dataset_id}/export")
        assert get_resp.status_code == 200
        assert get_resp.headers["content-type"] == "application/zip"
