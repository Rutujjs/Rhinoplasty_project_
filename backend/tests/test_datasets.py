from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_create_dataset():
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

    data = response.json()

    assert data["dataset_name"] == "Test_Dataset"
    assert data["status"] == "CREATED"
    assert data["generated_count"] == 0
    assert data["generation_attempts"] == 0


def test_get_dataset():
    response = client.get("/datasets/1")

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == 1
    assert data["dataset_name"] == "APAC_Facial_01"


def test_get_dataset_not_found():
    response = client.get("/datasets/999999")

    assert response.status_code == 404

    data = response.json()

    assert data["error"] == "Dataset not found"