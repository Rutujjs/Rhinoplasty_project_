from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_generate_dataset():
    # Create a fresh dataset for the test
    create_response = client.post(
        "/datasets/",
        json={
            "dataset_name": "Generation_Test",
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

    assert create_response.status_code == 200

    dataset_id = create_response.json()["id"]

    # Generate the requested image
    response = client.post(
        f"/datasets/{dataset_id}/generate"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["dataset_id"] == dataset_id
    assert data[0]["generation_model"] == "MOCK"
    assert data[0]["status"] == "GENERATED"
    assert data[0]["image_url"].startswith("/images/")

def test_generation_status():
    # Create a fresh dataset
    create_response = client.post(
        "/datasets/",
        json={
            "dataset_name": "Status_Test",
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

    assert create_response.status_code == 200

    dataset_id = create_response.json()["id"]

    # Generate the image
    generate_response = client.post(
        f"/datasets/{dataset_id}/generate"
    )

    assert generate_response.status_code == 200

    # Check generation status
    status_response = client.get(
        f"/datasets/{dataset_id}/generation-status"
    )

    assert status_response.status_code == 200

    data = status_response.json()

    assert data["dataset_id"] == dataset_id
    assert data["status"] == "COMPLETED"
    assert data["requested_images"] == 1
    assert data["generated_count"] == 1
    assert data["generation_attempts"] == 1
    assert data["progress_percentage"] == 100

def test_generation_failure():
    from unittest.mock import patch

    create_response = client.post(
        "/datasets/",
        json={
            "dataset_name": "Failure_Test",
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

    assert create_response.status_code == 200

    dataset_id = create_response.json()["id"]

    with patch(
        "backend.app.services.generation_service.generator.generate",
        side_effect=RuntimeError("Test generation failure"),
    ):
        response = client.post(
            f"/datasets/{dataset_id}/generate"
        )

    assert response.status_code == 500

    data = response.json()

    assert data["error"] == "Image generation failed"

    status_response = client.get(
        f"/datasets/{dataset_id}/generation-status"
    )

    assert status_response.status_code == 200

    status_data = status_response.json()

    assert status_data["status"] == "FAILED"
    assert status_data["generated_count"] == 0
    assert status_data["generation_attempts"] == 1
    assert status_data["progress_percentage"] == 0