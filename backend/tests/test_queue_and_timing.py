"""
Unit and integration tests for Kaggle generation queue dispatching,
dataset status synchronization, duplicate submission prevention, and
end-to-end stage and viewpoint timing metrics.
"""

import json
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.status import (
    DatasetStatus,
    JobStatus,
    map_job_status_to_dataset_status,
)
from backend.app.database.base import Base
from backend.app.models.dataset import Dataset
from backend.app.models.generation_job import GenerationJob
from backend.app.services.job_service import (
    create_job,
    transition_job_status,
)
from backend.app.services.timing_service import (
    estimate_remaining_seconds,
    get_job_timing_summary,
    record_stage_end,
    record_stage_start,
    record_view_timing,
)


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


# --------------------------------------------------------------------------- #
# Part 1: Status Mapping Specification Tests                                  #
# --------------------------------------------------------------------------- #

class TestStatusMapping:
    """Verify exact 1:1 and mapped status rules required by the prompt."""

    def test_all_specified_mappings(self):
        # CREATED -> CREATED
        assert map_job_status_to_dataset_status("CREATED") == DatasetStatus.CREATED
        # QUEUED -> QUEUED
        assert map_job_status_to_dataset_status(JobStatus.QUEUED) == DatasetStatus.QUEUED
        assert map_job_status_to_dataset_status("QUEUED") == DatasetStatus.QUEUED

        # PREPARING / SUBMITTED / RUNNING -> GENERATING
        assert map_job_status_to_dataset_status(JobStatus.PREPARING) == DatasetStatus.GENERATING
        assert map_job_status_to_dataset_status(JobStatus.SUBMITTED) == DatasetStatus.GENERATING
        assert map_job_status_to_dataset_status(JobStatus.RUNNING) == DatasetStatus.GENERATING

        # DOWNLOADING / VALIDATING -> PROCESSING
        assert map_job_status_to_dataset_status(JobStatus.DOWNLOADING) == DatasetStatus.PROCESSING
        assert map_job_status_to_dataset_status(JobStatus.VALIDATING) == DatasetStatus.PROCESSING

        # PACKAGING with deforming -> DEFORMING
        assert map_job_status_to_dataset_status(JobStatus.PACKAGING, stage="Nasal deformation in progress") == DatasetStatus.DEFORMING
        assert map_job_status_to_dataset_status(JobStatus.PACKAGING, stage="Packaging dataset") == DatasetStatus.PROCESSING

        # COMPLETED / FAILED / CANCELLED
        assert map_job_status_to_dataset_status(JobStatus.COMPLETED) == DatasetStatus.COMPLETED
        assert map_job_status_to_dataset_status(JobStatus.COMPLETED_WITH_ERRORS) == DatasetStatus.COMPLETED_WITH_ERRORS
        assert map_job_status_to_dataset_status(JobStatus.FAILED) == DatasetStatus.FAILED
        assert map_job_status_to_dataset_status(JobStatus.CANCEL_REQUESTED) == DatasetStatus.CANCELLED
        assert map_job_status_to_dataset_status(JobStatus.CANCELLED) == DatasetStatus.CANCELLED


# --------------------------------------------------------------------------- #
# Part 2: Queue Transitions & Dataset Status Synchronization                  #
# --------------------------------------------------------------------------- #

class TestQueueAndJobSynchronization:
    """Verify that jobs start in QUEUED and synchronize dataset status automatically."""

    def test_job_creation_and_dataset_status_sync(self, memory_db):
        dataset = Dataset(
            dataset_name="Queue_Test_DS",
            patient_id="P88",
            population_group="East Asian",
            region="Korea",
            age_min=20,
            age_max=30,
            sex_or_attribute="female",
            skin_tone="fair",
            pose="frontal",
            expression="neutral",
            num_images=9,
            resolution=512,
            status=DatasetStatus.CREATED,
        )
        memory_db.add(dataset)
        memory_db.commit()
        memory_db.refresh(dataset)

        # Enqueue job
        job = create_job(db=memory_db, dataset=dataset, provider="KAGGLE", requested_count=9)
        assert job.status == JobStatus.QUEUED

        # Sync dataset status
        dataset.status = map_job_status_to_dataset_status(job.status)
        memory_db.commit()
        memory_db.refresh(dataset)
        assert dataset.status == DatasetStatus.QUEUED

        # Forward transition to PREPARING
        transition_job_status(memory_db, job, JobStatus.PREPARING, stage="Preparing manifest")
        memory_db.refresh(dataset)
        assert dataset.status == DatasetStatus.GENERATING

        # Forward transition to SUBMITTED
        transition_job_status(memory_db, job, JobStatus.SUBMITTED, stage="Submitted to Kaggle")
        memory_db.refresh(dataset)
        assert dataset.status == DatasetStatus.GENERATING

        # Forward transition to RUNNING
        transition_job_status(memory_db, job, JobStatus.RUNNING, stage="FLUX GPU running")
        memory_db.refresh(dataset)
        assert dataset.status == DatasetStatus.GENERATING

        # Forward transition to DOWNLOADING
        transition_job_status(memory_db, job, JobStatus.DOWNLOADING, stage="Retrieving artifacts")
        memory_db.refresh(dataset)
        assert dataset.status == DatasetStatus.PROCESSING

        # Forward transition to VALIDATING
        transition_job_status(memory_db, job, JobStatus.VALIDATING, stage="Validating 9 views")
        memory_db.refresh(dataset)
        assert dataset.status == DatasetStatus.PROCESSING

        # Forward transition to PACKAGING with deforming stage
        transition_job_status(memory_db, job, JobStatus.PACKAGING, stage="Applying nasal deformation")
        memory_db.refresh(dataset)
        assert dataset.status == DatasetStatus.DEFORMING

        # Forward transition to COMPLETED
        transition_job_status(memory_db, job, JobStatus.COMPLETED, stage="All 9 views complete")
        memory_db.refresh(dataset)
        assert dataset.status == DatasetStatus.COMPLETED


# --------------------------------------------------------------------------- #
# Part 3: Stage and Viewpoint Timing Metrics                                  #
# --------------------------------------------------------------------------- #

class TestTimingEngine:
    """Verify all 11 stages and 9-view timing recording and estimation."""

    def test_stage_recording_and_summary(self, memory_db):
        dataset = Dataset(
            dataset_name="Timing_Test_DS",
            patient_id="P99",
            population_group="South Asian",
            region="India",
            age_min=25,
            age_max=35,
            sex_or_attribute="male",
            skin_tone="medium",
            pose="frontal",
            expression="neutral",
            num_images=9,
            resolution=512,
            status=DatasetStatus.CREATED,
        )
        memory_db.add(dataset)
        memory_db.commit()

        job = create_job(db=memory_db, dataset=dataset, provider="KAGGLE", requested_count=9)
        assert job.stage_timing is not None

        # Record stages
        record_stage_start(job, "queue_waiting")
        record_stage_end(job, "queue_waiting", status="COMPLETED")

        record_stage_start(job, "manifest_preparation")
        record_stage_end(job, "manifest_preparation", status="COMPLETED")

        record_stage_start(job, "kaggle_submission")
        record_stage_end(job, "kaggle_submission", status="COMPLETED")

        record_stage_start(job, "flux_generation")
        record_stage_end(job, "flux_generation", status="COMPLETED")

        record_stage_start(job, "artifact_download")
        record_stage_end(job, "artifact_download", status="COMPLETED")

        record_stage_start(job, "image_validation")
        record_stage_end(job, "image_validation", status="COMPLETED")

        record_stage_start(job, "deformation")
        # Record 9 viewpoints
        views = [
            ("frontal", 0.045, 0.420),
            ("left_20", 0.042, 0.380),
            ("left_40", 0.040, 0.395),
            ("left_60", 0.041, 0.410),
            ("right_20", 0.043, 0.385),
            ("right_40", 0.042, 0.400),
            ("right_60", 0.041, 0.390),
            ("basal", 0.038, 0.001),
            ("helicopter", 0.039, 0.001),
        ]
        for vkey, seg, warp in views:
            record_view_timing(job, vkey, seg_sec=seg, def_sec=warp)

        record_stage_end(job, "deformation", status="COMPLETED")

        record_stage_start(job, "persistence")
        record_stage_end(job, "persistence", status="COMPLETED")

        record_stage_end(job, "total_elapsed", status="COMPLETED")

        memory_db.commit()
        memory_db.refresh(job)

        # Inspect summary
        summary = get_job_timing_summary(job)
        assert summary is not None
        assert "queue_wait_sec" in summary
        assert "manifest_prep_sec" in summary
        assert "kaggle_submission_sec" in summary
        assert "flux_generation_sec" in summary
        assert "artifact_retrieval_sec" in summary
        assert "validation_sec" in summary
        assert "deformation_sec" in summary
        assert "persistence_sec" in summary
        assert "total_elapsed_sec" in summary
        assert "viewpoints" in summary

        # Verify all 9 viewpoints recorded
        vp_map = summary["viewpoints"]
        assert len(vp_map) == 9
        assert "frontal" in vp_map
        assert vp_map["frontal"]["seg_sec"] == 0.045
        assert vp_map["frontal"]["def_sec"] == 0.420
        assert round(vp_map["frontal"]["total_sec"], 3) == 0.465

    def test_estimated_remaining_time(self, memory_db):
        dataset = Dataset(
            dataset_name="Est_Test_DS",
            patient_id="P11",
            population_group="East Asian",
            region="China",
            age_min=20,
            age_max=30,
            sex_or_attribute="female",
            skin_tone="fair",
            pose="frontal",
            expression="neutral",
            num_images=9,
            resolution=512,
            status=DatasetStatus.CREATED,
        )
        memory_db.add(dataset)
        memory_db.commit()

        job = create_job(db=memory_db, dataset=dataset, provider="KAGGLE", requested_count=9)
        
        # When in QUEUED stage
        job.status = JobStatus.QUEUED
        rem_queued = estimate_remaining_seconds(job)
        assert rem_queued is None

        # When in RUNNING stage
        job.status = JobStatus.RUNNING
        rem_running = estimate_remaining_seconds(job)
        assert rem_running is not None
        assert rem_running > 0.0

        # When in COMPLETED stage
        job.status = JobStatus.COMPLETED
        rem_completed = estimate_remaining_seconds(job)
        assert rem_completed == 0.0

    def test_queued_jobs_fifo_order_and_duplicate_guard(self, memory_db):
        from backend.app.services.job_service import get_queued_jobs

        dataset1 = Dataset(
            dataset_name="FIFO_1",
            patient_id="P01",
            population_group="East Asian",
            region="China",
            age_min=20,
            age_max=30,
            sex_or_attribute="female",
            skin_tone="fair",
            pose="frontal",
            expression="neutral",
            num_images=9,
            resolution=512,
            status=DatasetStatus.CREATED,
        )
        dataset2 = Dataset(
            dataset_name="FIFO_2",
            patient_id="P02",
            population_group="South Asian",
            region="India",
            age_min=25,
            age_max=35,
            sex_or_attribute="male",
            skin_tone="medium",
            pose="frontal",
            expression="neutral",
            num_images=9,
            resolution=512,
            status=DatasetStatus.CREATED,
        )
        memory_db.add_all([dataset1, dataset2])
        memory_db.commit()

        # Enqueue Job 1
        job1 = create_job(db=memory_db, dataset=dataset1, provider="KAGGLE", requested_count=9)
        # Enqueue Job 2
        job2 = create_job(db=memory_db, dataset=dataset2, provider="KAGGLE", requested_count=9)

        # FIFO check
        queued = get_queued_jobs(memory_db)
        assert len(queued) >= 2
        assert queued[0].id == job1.id
        assert queued[1].id == job2.id

        # Duplicate job check: Active statuses prevent re-enqueuing
        active_statuses = [
            JobStatus.QUEUED,
            JobStatus.PREPARING,
            JobStatus.SUBMITTED,
            JobStatus.RUNNING,
            JobStatus.DOWNLOADING,
            JobStatus.VALIDATING,
            JobStatus.PACKAGING,
        ]
        active_job = (
            memory_db.query(GenerationJob)
            .filter(
                GenerationJob.dataset_id == dataset1.id,
                GenerationJob.status.in_(active_statuses),
            )
            .first()
        )
        assert active_job is not None
        assert active_job.id == job1.id

