import sys
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

# Add project root
sys.path.insert(0, ".")

import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET"] = "test-secret"
os.environ["GENERATION_PROVIDER"] = "MOCK"
os.environ["GENERATION_MODEL"] = "MOCK"
os.environ["RUN_EMBEDDED_WORKER"] = "false"

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database.session import SessionLocal, engine
from backend.app.database.base import Base
from backend.tests.conftest import create_test_dataset
from backend.app.models.dataset import Dataset
from backend.app.models.generation_job import GenerationJob
from backend.app.core.status import DatasetStatus, JobStatus
from backend.app.services.job_service import create_job
from backend.worker import _kaggle_prepare

Base.metadata.create_all(bind=engine)
client = TestClient(app)
db = SessionLocal()

print("Creating test dataset...")
dataset_dict = create_test_dataset(client, overrides={"num_images": 2})
dataset_id = dataset_dict["id"]
dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
dataset.status = DatasetStatus.GENERATING
db.flush()

print("Creating test job...")
job = create_job(db, dataset=dataset, provider="KAGGLE_BATCH", requested_count=2)
job.status = JobStatus.PREPARING.value
job.batch_id = "test-batch-xyz"
db.commit()
db.refresh(job)
db.refresh(dataset)

mock_provider = MagicMock()
mock_provider.prepare_batch.return_value = MagicMock(succeeded=True, message="OK")
mock_provider.submit_batch.return_value = MagicMock(
    succeeded=True,
    provider_run_id="user/apac-gen",
    message="OK",
)

tmp_artifacts = Path(tempfile.gettempdir()) / "apac_test_artifacts"

print("Calling _kaggle_prepare...")
with patch("backend.worker.get_provider", return_value=mock_provider), \
     patch("backend.worker.settings") as s:
    s.ARTIFACT_DOWNLOAD_DIR = str(tmp_artifacts)
    s.IMAGE_WIDTH = 512
    s.IMAGE_HEIGHT = 512
    s.KAGGLE_POST_PUSH_DELAY_SECONDS = 90
    s.JOB_LEASE_SECONDS = 300
    _kaggle_prepare(db, job, dataset)

print("Finished _kaggle_prepare successfully!")
