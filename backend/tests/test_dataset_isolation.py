"""
Integration and Regression Tests for Multi-Dataset Data Isolation and Storage.

Verifies:
  1. Two datasets generated with the same patient_id (P01) have separate isolated paths.
  2. Empty datasets never inherit or steal images from unrelated datasets with the same patient_id.
  3. Real SQLite database (storage/apac_face.db) has isolated paths and distinct images for Datasets 7, 8, 9.
"""

import hashlib
import sqlite3
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage
from backend.app.services.deformation_service import deformation_service
from backend.tests.conftest import create_test_dataset


def _compute_sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _run_worker_once(db_session: Session):
    """Run one mock worker cycle using the test DB session."""
    from backend.app.services.job_service import get_queued_jobs, claim_job
    from backend.worker import _process_job, WORKER_ID

    jobs = get_queued_jobs(db_session, limit=10)
    for job in jobs:
        claim_job(db_session, job, worker_id=WORKER_ID)
        _process_job(db_session, job)


class TestDatasetDataIsolation:
    def test_two_datasets_same_patient_id_isolated_storage(self, client: TestClient, db_session: Session):
        """Generating two datasets with patient_id P01 produces separate, isolated storage paths."""
        ds1 = create_test_dataset(client, overrides={"dataset_name": "DS_One", "num_images": 9, "patient_id": "P01"})
        ds2 = create_test_dataset(client, overrides={"dataset_name": "DS_Two", "num_images": 9, "patient_id": "P01"})

        ds1_id = ds1["id"]
        ds2_id = ds2["id"]

        # Generate images for both
        client.post(f"/datasets/{ds1_id}/generate")
        _run_worker_once(db_session)

        client.post(f"/datasets/{ds2_id}/generate")
        _run_worker_once(db_session)

        # Query database for both datasets
        imgs_1 = db_session.query(GeneratedImage).filter(GeneratedImage.dataset_id == ds1_id).all()
        imgs_2 = db_session.query(GeneratedImage).filter(GeneratedImage.dataset_id == ds2_id).all()

        assert len(imgs_1) == 9
        assert len(imgs_2) == 9

        for img in imgs_1:
            assert f"dataset_{ds1_id}" in img.image_path, f"DS1 path missing dataset_{ds1_id}: {img.image_path}"
            assert Path(img.image_path).exists()

        for img in imgs_2:
            assert f"dataset_{ds2_id}" in img.image_path, f"DS2 path missing dataset_{ds2_id}: {img.image_path}"
            assert Path(img.image_path).exists()

    def test_ensure_dataset_images_never_steals_from_other_datasets(self, db_session: Session):
        """A new dataset with patient_id P01 must NOT inherit images from existing P01 datasets."""
        new_ds = Dataset(
            dataset_name="Test_Isolation_Isolated_DS",
            patient_id="P01",
            region="Japan",
            population_group="East Asian",
            age_min=20,
            age_max=30,
            sex_or_attribute="female",
            skin_tone="fair",
            pose="frontal",
            expression="neutral",
            num_images=9,
            resolution=512,
            status="CREATED",
            generated_count=0,
        )
        db_session.add(new_ds)
        db_session.commit()

        imgs = deformation_service._ensure_dataset_images(db_session, new_ds, patient_id="P01")
        assert len(imgs) == 0, f"Expected 0 images for ungenerated dataset, got {len(imgs)}"
        assert new_ds.generated_count == 0

    def test_real_database_datasets_7_8_9_isolation(self):
        """Verify on-disk real SQLite database storage/apac_face.db has isolated paths and distinct images."""
        db_path = Path("storage/apac_face.db")
        if not db_path.exists():
            pytest.skip("storage/apac_face.db does not exist")

        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()

        # Check datasets 7, 8, 9 have 9 viewpoints each
        for ds_id in [7, 8, 9]:
            c.execute("SELECT COUNT(*) FROM generated_images WHERE dataset_id = ?", (ds_id,))
            cnt = c.fetchone()[0]
            assert cnt == 9, f"Dataset {ds_id} must have 9 viewpoints in storage/apac_face.db"

        # Check path isolation
        for ds_id in [7, 8, 9]:
            c.execute("SELECT image_path FROM generated_images WHERE dataset_id = ?", (ds_id,))
            paths = [r[0] for r in c.fetchall()]
            for p in paths:
                assert f"dataset_{ds_id}" in p, f"Dataset {ds_id} path {p} not isolated"
                assert Path(p).exists(), f"File {p} must exist on disk"

        # Check distinct hashes across all viewpoints for Datasets 7, 8, 9
        viewpoints = ["frontal", "left_20", "left_40", "left_60", "right_20", "right_40", "right_60", "basal", "helicopter"]
        for vp in viewpoints:
            hashes = {}
            for ds_id in [7, 8, 9]:
                c.execute("SELECT image_path FROM generated_images WHERE dataset_id = ? AND viewpoint = ?", (ds_id, vp))
                row = c.fetchone()
                assert row is not None, f"Dataset {ds_id} missing {vp}"
                p = Path(row[0])
                assert p.exists()
                hashes[ds_id] = _compute_sha256(p.read_bytes())

            assert hashes[7] != hashes[8], f"Viewpoint {vp}: DS7 and DS8 must be distinct"
            assert hashes[7] != hashes[9], f"Viewpoint {vp}: DS7 and DS9 must be distinct"
            assert hashes[8] != hashes[9], f"Viewpoint {vp}: DS8 and DS9 must be distinct"

        conn.close()
