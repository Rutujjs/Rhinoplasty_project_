"""
Tests for generation endpoints.

The generation route is ASYNCHRONOUS:
  POST /datasets/{id}/generate
    → HTTP 202 Accepted
    → body: JobSubmitResponse { job_id, dataset_id, status, provider, status_url, message }

  GET /datasets/{id}/generation-status
    → HTTP 200
    → body: GenerationStatusResponse { dataset_id, status, requested_images,
                                        generated_count, generation_attempts,
                                        progress_percentage }

These tests use the isolated in-memory SQLite database provided by
conftest.py – they do not touch the real PostgreSQL database and do not
make network calls to Kaggle.

The actual image generation is performed by the background worker; the
API layer only enqueues the job.  Worker integration is tested separately.
"""
from fastapi.testclient import TestClient

from backend.tests.conftest import create_test_dataset


class TestEnqueueGeneration:
    """POST /datasets/{id}/generate → 202 with job details."""

    def test_enqueue_returns_202(self, client: TestClient):
        dataset = create_test_dataset(client)
        response = client.post(f"/datasets/{dataset['id']}/generate")
        assert response.status_code == 202

    def test_enqueue_response_has_job_id(self, client: TestClient):
        dataset = create_test_dataset(client)
        response = client.post(f"/datasets/{dataset['id']}/generate")
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], int)

    def test_enqueue_response_fields(self, client: TestClient):
        dataset = create_test_dataset(client)
        response = client.post(f"/datasets/{dataset['id']}/generate")
        assert response.status_code == 202
        data = response.json()
        assert data["dataset_id"] == dataset["id"]
        assert data["status"] == "QUEUED"
        assert data["provider"] == "MOCK"
        assert "/generation-jobs/" in data["status_url"]
        assert "message" in data

    def test_enqueue_nonexistent_dataset_returns_404(self, client: TestClient):
        response = client.post("/datasets/999999/generate")
        assert response.status_code == 404
        assert response.json()["error"] == "Dataset not found"

    def test_enqueue_already_generating_returns_409(self, client: TestClient):
        """Second enqueue while first is GENERATING must return 409."""
        from unittest.mock import patch
        from backend.app.core.status import DatasetStatus

        dataset = create_test_dataset(client)
        dataset_id = dataset["id"]

        with patch(
            "backend.app.api.routes.generation.get_dataset_by_id",
        ) as mock_get:
            mock_ds = mock_get.return_value
            mock_ds.status = DatasetStatus.GENERATING
            mock_ds.id = dataset_id

            response = client.post(f"/datasets/{dataset_id}/generate")

        assert response.status_code == 409
        assert "already in progress" in response.json()["error"]

    def test_enqueue_completed_dataset_returns_409(self, client: TestClient):
        from unittest.mock import patch
        from backend.app.core.status import DatasetStatus

        dataset = create_test_dataset(client)
        dataset_id = dataset["id"]

        with patch(
            "backend.app.api.routes.generation.get_dataset_by_id",
        ) as mock_get:
            mock_ds = mock_get.return_value
            mock_ds.status = DatasetStatus.COMPLETED
            mock_ds.id = dataset_id

            response = client.post(f"/datasets/{dataset_id}/generate")

        assert response.status_code == 409
        assert "already been generated" in response.json()["error"]

    def test_enqueue_twice_creates_two_jobs(self, client: TestClient):
        """Re-enqueueing a CREATED dataset a second time is only blocked when
        status is GENERATING or COMPLETED.  Here we verify unique job IDs."""
        dataset = create_test_dataset(client)
        r1 = client.post(f"/datasets/{dataset['id']}/generate")
        assert r1.status_code == 202
        job_id_1 = r1.json()["job_id"]

        # Reset dataset to CREATED so second enqueue is allowed.
        from backend.app.models.dataset import Dataset
        from backend.app.core.status import DatasetStatus

        # Use the db_session through the override – reach it via a GET then
        # directly manipulate via the dependency override stored in conftest.
        # Simpler: just patch get_dataset_by_id to return CREATED status.
        # Actually the simplest is to create a second fresh dataset.
        dataset2 = create_test_dataset(client, {"dataset_name": "DS_Second"})
        r2 = client.post(f"/datasets/{dataset2['id']}/generate")
        assert r2.status_code == 202
        job_id_2 = r2.json()["job_id"]

        assert job_id_1 != job_id_2


class TestGenerationStatus:
    """GET /datasets/{id}/generation-status – backward-compatible endpoint."""

    def test_status_before_generation(self, client: TestClient):
        dataset = create_test_dataset(client)
        response = client.get(f"/datasets/{dataset['id']}/generation-status")
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] == dataset["id"]
        assert data["status"] == "CREATED"
        assert data["generated_count"] == 0
        assert data["progress_percentage"] == 0.0

    def test_status_after_enqueue(self, client: TestClient):
        """After enqueueing, dataset status should be GENERATING."""
        dataset = create_test_dataset(client)
        client.post(f"/datasets/{dataset['id']}/generate")

        response = client.get(f"/datasets/{dataset['id']}/generation-status")
        assert response.status_code == 200
        data = response.json()
        # After enqueue the dataset status is set to GENERATING.
        assert data["status"] == "GENERATING"
        assert data["requested_images"] == 1

    def test_status_nonexistent_dataset_returns_404(self, client: TestClient):
        response = client.get("/datasets/999999/generation-status")
        assert response.status_code == 404

    def test_status_progress_percentage_field_present(self, client: TestClient):
        dataset = create_test_dataset(client)
        response = client.get(f"/datasets/{dataset['id']}/generation-status")
        data = response.json()
        assert "progress_percentage" in data
        assert isinstance(data["progress_percentage"], float)

    def test_status_requested_images_matches_dataset(self, client: TestClient):
        dataset = create_test_dataset(client, {"num_images": 5})
        response = client.get(f"/datasets/{dataset['id']}/generation-status")
        data = response.json()
        assert data["requested_images"] == 5