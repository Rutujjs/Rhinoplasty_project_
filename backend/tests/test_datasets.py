"""
Tests for the /datasets/ endpoints.

These tests use the isolated in-memory SQLite database provided by
conftest.py – they do not touch the real PostgreSQL database.
"""
from fastapi.testclient import TestClient

from backend.tests.conftest import create_test_dataset


class TestCreateDataset:
    def test_create_returns_200(self, client: TestClient):
        response = client.post(
            "/datasets/",
            json={
                "dataset_name": "Test_Dataset",
                "region": "India",
                "population_group": "South Asian",
                "age_min": 25,
                "age_max": 35,
                "sex_or_attribute": "female",
                "skin_tone": "medium_brown",
                "pose": "frontal",
                "expression": "neutral",
                "medical_condition": None,
                "medical_severity": None,
                "num_images": 1,
                "resolution": 512,
            },
        )
        assert response.status_code == 200

    def test_create_response_fields(self, client: TestClient):
        data = create_test_dataset(client)
        assert data["dataset_name"] == "Test_Dataset"
        assert data["status"] == "CREATED"
        assert data["generated_count"] == 0
        assert data["generation_attempts"] == 0

    def test_create_assigns_id(self, client: TestClient):
        data = create_test_dataset(client)
        assert "id" in data
        assert isinstance(data["id"], int)

    def test_age_min_greater_than_age_max_rejected(self, client: TestClient):
        response = client.post(
            "/datasets/",
            json={
                "dataset_name": "Bad_Age",
                "region": "India",
                "population_group": "South Asian",
                "age_min": 40,
                "age_max": 30,          # invalid: min > max
                "sex_or_attribute": "female",
                "skin_tone": "medium_brown",
                "pose": "frontal",
                "expression": "neutral",
                "num_images": 1,
                "resolution": 512,
            },
        )
        assert response.status_code == 422

    def test_create_multiple_datasets(self, client: TestClient):
        d1 = create_test_dataset(client, {"dataset_name": "DS_Alpha"})
        d2 = create_test_dataset(client, {"dataset_name": "DS_Beta"})
        assert d1["id"] != d2["id"]


class TestGetDataset:
    def test_get_existing_dataset(self, client: TestClient):
        created = create_test_dataset(client)
        dataset_id = created["id"]

        response = client.get(f"/datasets/{dataset_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == dataset_id
        assert data["dataset_name"] == "Test_Dataset"

    def test_get_nonexistent_dataset_returns_404(self, client: TestClient):
        response = client.get("/datasets/999999")
        assert response.status_code == 404

        data = response.json()
        assert data["error"] == "Dataset not found"


class TestListDatasets:
    def test_list_empty(self, client: TestClient):
        response = client.get("/datasets/")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_returns_created_datasets(self, client: TestClient):
        create_test_dataset(client, {"dataset_name": "DS_1"})
        create_test_dataset(client, {"dataset_name": "DS_2"})

        response = client.get("/datasets/")
        assert response.status_code == 200

        names = [d["dataset_name"] for d in response.json()]
        assert "DS_1" in names
        assert "DS_2" in names