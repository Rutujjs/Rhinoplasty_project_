"""
Shared pytest fixtures for the backend test suite.

Key design decisions
--------------------
1.  Tests use an **in-memory SQLite** database so they are fast, isolated,
    and never touch the real PostgreSQL database.

2.  The `client` fixture creates a fresh database schema for every test
    function and tears it down afterwards.  Tests cannot accidentally share
    state through the database.

3.  The FastAPI dependency `get_db` is overridden via `app.dependency_overrides`
    so every route that calls `Depends(get_db)` receives the test database
    session automatically.

4.  Environment variables that differ from production defaults are patched
    at the module level using `monkeypatch` where individual tests need them,
    or via the `TEST_DATABASE_URL` constant here.

Usage
-----
Any test file can import these fixtures simply by declaring them as
function parameters – pytest discovers conftest.py automatically.

    def test_something(client):
        response = client.get("/health")
        assert response.status_code == 200
"""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker, Session

# Force in-memory SQLite before settings / app import
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET"] = "test-secret-not-for-production"

# --------------------------------------------------------------------------- #
# SQLite engine – created once per test session for speed.                     #
# StaticPool + same_thread=False are required for in-memory SQLite + pytest.   #
# --------------------------------------------------------------------------- #
TEST_DB_URL = "sqlite:///:memory:"

_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

_TestingSessionLocal = sessionmaker(
    bind=_engine,
    autoflush=False,
    autocommit=False,
)

# Patch connection and session engines before importing app
import backend.app.database.connection as db_conn
import backend.app.database.session as db_sess

db_conn.engine = _engine
db_sess.engine = _engine
db_sess.SessionLocal = _TestingSessionLocal

from backend.app.database.base import Base
from backend.app.database.session import get_db
from backend.app.main import app

# Import all models so SQLAlchemy registers their tables with Base.metadata.
import backend.app.models  # noqa: F401


def _create_schema():
    """Create all tables in the in-memory database."""
    Base.metadata.create_all(bind=_engine)


def _drop_schema():
    """Drop all tables so the next test starts clean."""
    Base.metadata.drop_all(bind=_engine)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def db_session() -> Session:
    """
    Yield a SQLAlchemy session backed by the in-memory SQLite test database.

    Schema is recreated for every test function to ensure isolation.
    """
    _create_schema()
    session = _TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        _drop_schema()


@pytest.fixture()
def client(db_session: Session) -> TestClient:
    """
    Yield a FastAPI TestClient that uses the isolated test database.

    The `get_db` dependency is overridden so that every route handler
    receives `db_session` instead of the production database session.
    """
    def override_get_db():
        try:
            yield db_session
        finally:
            pass  # session lifecycle managed by db_session fixture

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    # Remove the override after the test so other tests are unaffected.
    app.dependency_overrides.pop(get_db, None)


# --------------------------------------------------------------------------- #
# Helper factories (used by multiple test files)                               #
# --------------------------------------------------------------------------- #

DATASET_PAYLOAD = {
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
}


def create_test_dataset(client: TestClient, overrides: dict | None = None) -> dict:
    """
    POST a dataset and return the parsed JSON response.

    Pass `overrides` to change specific fields from the default payload.
    """
    payload = {**DATASET_PAYLOAD, **(overrides or {})}
    response = client.post("/datasets/", json=payload)
    assert response.status_code == 200, (
        f"create_test_dataset failed: {response.status_code} {response.text}"
    )
    return response.json()
