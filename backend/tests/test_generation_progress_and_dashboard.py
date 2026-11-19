"""
Tests for generation progress tracking, MOCK worker multi-view execution,
and dashboard stats endpoints.
"""

from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.core.status import DatasetStatus, JobStatus
from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage
from backend.app.models.generation_job import GenerationJob
from backend.app.services.job_service import (
    claim_job,
    create_job,
    get_job_by_id,
    transition_job_status,
)
from backend.worker import _process_job, WORKER_ID
from backend.tests.conftest import create_test_dataset


def test_dashboard_stats_endpoint_success(client: TestClient, db_session: Session):
    """GET /dashboard/stats must return 200 without AttributeError on j.stage."""
    # Create a test dataset and job
    ds = create_test_dataset(client)
    job_resp = client.post(f"/datasets/{ds['id']}/generate")
    assert job_resp.status_code == 202

    resp = client.get("/dashboard/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_datasets" in data
    assert "total_generated_images" in data
    assert "recent_jobs" in data
    assert len(data["recent_jobs"]) >= 1
    job_data = data["recent_jobs"][0]
    assert "stage" in job_data
    assert job_data["stage"] is not None


def test_mock_worker_generates_all_nine_viewpoints_and_progress(
    client: TestClient,
    db_session: Session,
):
    """
    Worker must generate all 9 clinical viewpoints for a 9-view dataset,
    save valid image files to disk, create 9 GeneratedImage DB records,
    and reach 100% progress with JobStatus.COMPLETED.
    """
    ds = create_test_dataset(client, overrides={"num_images": 9})
    ds_id = ds["id"]

    # Enqueue generation job
    enqueue_resp = client.post(f"/datasets/{ds_id}/generate")
    assert enqueue_resp.status_code == 202
    job_id = enqueue_resp.json()["job_id"]

    # Run the worker cycle
    job = get_job_by_id(db_session, job_id)
    assert job is not None
    assert job.status == JobStatus.QUEUED
    assert job.progress_percentage == 0.0

    # Claim and execute job
    claim_job(db_session, job, worker_id=WORKER_ID)
    _process_job(db_session, job)

    # Reload from DB
    db_session.refresh(job)
    dataset = db_session.query(Dataset).filter(Dataset.id == ds_id).first()
    assert dataset is not None

    # Check terminal status and counters
    assert job.status == JobStatus.COMPLETED
    assert job.progress_percentage == 100.0
    assert job.generated_count == 9
    assert job.accepted_count == 9
    assert job.failed_count == 0
    assert dataset.status == DatasetStatus.COMPLETED
    assert dataset.generated_count == 9

    # Check that 9 GeneratedImage records were created
    images = (
        db_session.query(GeneratedImage)
        .filter(GeneratedImage.job_id == job_id)
        .all()
    )
    assert len(images) == 9

    # Verify viewpoints are all distinct and match clinical sequence
    viewpoints = [img.viewpoint for img in images]
    assert "frontal" in viewpoints
    assert "left_20" in viewpoints
    assert "left_40" in viewpoints
    assert "left_60" in viewpoints
    assert "right_20" in viewpoints
    assert "right_40" in viewpoints
    assert "right_60" in viewpoints
    assert "basal" in viewpoints
    assert "helicopter" in viewpoints

    # Check that image files exist on disk
    for img in images:
        p = Path(img.image_path)
        assert p.exists()
        assert p.stat().st_size > 0


def test_mock_worker_handles_cancellation_mid_generation(
    client: TestClient,
    db_session: Session,
):
    """If a job status is CANCEL_REQUESTED, worker transitions to CANCELLED."""
    ds = create_test_dataset(client, overrides={"num_images": 9})
    ds_id = ds["id"]

    enqueue_resp = client.post(f"/datasets/{ds_id}/generate")
    job_id = enqueue_resp.json()["job_id"]

    job = get_job_by_id(db_session, job_id)
    assert job is not None
    claim_job(db_session, job, worker_id=WORKER_ID)

    # Request cancellation
    transition_job_status(db_session, job, JobStatus.CANCEL_REQUESTED)

    # Process job
    _process_job(db_session, job)

    db_session.refresh(job)
    assert job.status == JobStatus.CANCELLED


def test_job_retry_flow(client: TestClient, db_session: Session):
    """A failed job can be retried and reset to QUEUED."""
    ds = create_test_dataset(client, overrides={"num_images": 9})
    ds_id = ds["id"]

    enqueue_resp = client.post(f"/datasets/{ds_id}/generate")
    job_id = enqueue_resp.json()["job_id"]

    job = get_job_by_id(db_session, job_id)
    claim_job(db_session, job, worker_id=WORKER_ID)
    transition_job_status(db_session, job, JobStatus.FAILED, error_summary="Simulated test error")

    # Retry endpoint
    retry_resp = client.post(f"/generation-jobs/{job_id}/retry")
    assert retry_resp.status_code == 200
    assert retry_resp.json()["status"] == "QUEUED"
    assert retry_resp.json()["retry_count"] == 1
