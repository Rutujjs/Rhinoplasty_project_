import sys
from pathlib import Path
import asyncio

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, ".")

import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET"] = "test-secret"
os.environ["GENERATION_PROVIDER"] = "MOCK"
os.environ["GENERATION_MODEL"] = "MOCK"
os.environ["RUN_EMBEDDED_WORKER"] = "false"

from backend.tests.test_kaggle_provider import TestKaggleWorkerPath
from backend.tests.conftest import client, db_session, _engine, _TestingSessionLocal
from fastapi.testclient import TestClient
from backend.app.main import app

sess = _TestingSessionLocal()
test_obj = TestKaggleWorkerPath()

print("Setting up client and db...")
with TestClient(app) as cl:
    print("Running test...")
    test_obj.test_kaggle_prepare_succeeds_and_transitions_to_submitted(cl, sess)
    print("Test passed!")
