# Developer Guide

This guide covers setting up the development environment, running tests, extending the system, and working with the database.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Running the Application](#running-the-application)
- [Running Tests](#running-tests)
- [Adding a New Generation Provider](#adding-a-new-generation-provider)
- [Adding a New API Route](#adding-a-new-api-route)
- [Database Migrations](#database-migrations)
- [Troubleshooting Development Issues](#troubleshooting-development-issues)

---

## Prerequisites

| Tool | Minimum version | Notes |
|---|---|---|
| Python | 3.11+ | 3.12 recommended |
| Git | Any | For version control |
| Kaggle account | — | Required for `KAGGLE_BATCH` provider only |

---

## Environment Setup

### 1. Create the virtual environment

```powershell
cd c:\Users\poter\Downloads\synthetic
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

> [!IMPORTANT]
> **Always use the project `.venv`**, not the system Python. The system may have `kaggle 2.x` installed, which causes HTTP 401 errors due to API changes. The project requires `kaggle 1.7.x`.

### 3. Configure environment

```powershell
copy .env.example .env   # if .env.example exists, otherwise create .env manually
```

For local development, set:

```env
GENERATION_PROVIDER=MOCK
DATABASE_URL=sqlite:///./storage/apac_face.db
RUN_EMBEDDED_WORKER=true
```

See [README.md](../README.md#environment-configuration) for all settings.

### 4. Verify setup

```powershell
python diagnostic_suite.py
```

All `LOCAL` checks should pass before continuing.

---

## Project Structure

```
synthetic/
├── backend/
│   ├── app/
│   │   ├── api/routes/          # FastAPI route handlers
│   │   ├── core/
│   │   │   ├── config.py        # Settings (pydantic-settings)
│   │   │   └── status.py        # State machine enums
│   │   ├── models/              # SQLAlchemy ORM models
│   │   ├── schemas/             # Pydantic request/response schemas
│   │   ├── services/            # Business logic layer
│   │   └── main.py              # FastAPI app, routers, lifespan
│   ├── tests/                   # pytest test suite
│   └── worker.py                # Background job processor
├── deformation/
│   ├── engine.py                # NasalDeformationEngine
│   └── algorithms/              # Per-morphology algorithm implementations
├── generation/
│   ├── interface.py             # FaceGenerator ABC
│   ├── manifest.py              # GenerationManifest model
│   ├── kaggle_provider.py       # KaggleBatchProvider
│   ├── mock_provider.py         # MockFaceGenerator
│   └── viewpoints.py            # 9 canonical clinical viewpoints
├── validation/
│   └── pipeline.py              # Image validation pipeline
├── storage/                     # Runtime storage (git-ignored)
├── docs/                        # This documentation
├── diagnostic_suite.py          # End-to-end diagnostic tool
├── start_app.cmd                # Windows launcher
└── .env                         # Local configuration (git-ignored)
```

---

## Running the Application

### Windows (recommended)

```powershell
.\start_app.cmd
```

### Manual

```powershell
.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Running Tests

### Full test suite

```powershell
.venv\Scripts\Activate.ps1
pytest backend/tests/ -v
```

### Run a specific test file

```powershell
pytest backend/tests/test_generation.py -v
```

### Run with coverage

```powershell
pytest backend/tests/ --cov=backend --cov-report=term-missing
```

### Test environment behaviour

The embedded background worker is **automatically suppressed** during pytest runs. The worker checks for `PYTEST_CURRENT_TEST` in the environment and skips starting the daemon thread. This means jobs submitted during tests will stay in `QUEUED` state unless you call `_process_job()` directly in your test.

### Writing a new test

```python
# backend/tests/test_my_feature.py
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
```

---

## Adding a New Generation Provider

The generation system uses an abstract base class (`FaceGenerator`) in `generation/interface.py`. To add a new provider:

### 1. Create the provider module

```python
# generation/my_provider.py
from generation.interface import FaceGenerator, ProviderOperationResult, ProviderOutcome
from generation.manifest import GenerationManifest

class MyProvider(FaceGenerator):
    def generate_batch(self, manifest: GenerationManifest) -> ProviderOperationResult:
        # Implement batch generation
        ...
        return ProviderOperationResult(
            outcome=ProviderOutcome.SUCCESS,
            images=[...],   # list of downloaded image paths
        )

    def get_status(self, job_id: str) -> dict:
        # Return current status of a submitted job
        ...

    def cancel(self, job_id: str) -> bool:
        # Cancel a running job; return True if cancelled
        ...
```

### 2. Register in config

Add the provider name to `GENERATION_PROVIDER` in `backend/app/core/config.py`:

```python
GENERATION_PROVIDER: Literal["MOCK", "KAGGLE_BATCH", "MY_PROVIDER"] = "MOCK"
```

### 3. Wire in the worker

In `backend/worker.py`, add a branch in `_process_job()`:

```python
elif provider == "MY_PROVIDER":
    _process_my_provider_job(db, job, dataset)
```

Implement `_process_my_provider_job()` following the same pattern as `_process_mock_job()`.

### 4. Add a provider factory

In `backend/app/services/generation_service.py` (or a new factory module), add your provider to the factory function so it's instantiated correctly from the `GENERATION_PROVIDER` setting.

---

## Adding a New API Route

### 1. Create the router

```python
# backend/app/api/routes/my_feature.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.core.database import get_db

router = APIRouter(prefix="/my-feature", tags=["My Feature"])

@router.get("/")
def list_items(db: Session = Depends(get_db)):
    return {"items": []}
```

### 2. Register in main.py

```python
# backend/app/main.py
from backend.app.api.routes.my_feature import router as my_feature_router

app.include_router(my_feature_router, prefix="/api/v1")
```

### 3. Add Pydantic schemas

```python
# backend/app/schemas/my_feature.py
from pydantic import BaseModel

class MyFeatureResponse(BaseModel):
    id: int
    name: str
```

### 4. Write tests

```python
# backend/tests/test_my_feature.py
def test_list_items(client):
    resp = client.get("/api/v1/my-feature/")
    assert resp.status_code == 200
```

---

## Database Migrations

The project uses **SQLAlchemy with auto-create** (`Base.metadata.create_all(engine)`). There is no Alembic migration setup.

### Adding a new column

1. Add the column to the ORM model in `backend/app/models/`.
2. **Drop and recreate the database** (easiest approach for development):
   ```powershell
   Remove-Item storage\apac_face.db
   # Restart the app — it will auto-create the schema
   ```
3. Or add the column manually with SQLite tools if you need to preserve data.

> [!WARNING]
> SQLite does not support `DROP COLUMN`. If you need to remove a column in development, drop and recreate the database. For production data, export first.

### Inspecting the database

```powershell
# Using sqlite3 CLI
sqlite3 storage\apac_face.db ".tables"
sqlite3 storage\apac_face.db ".schema generation_jobs"
sqlite3 storage\apac_face.db "SELECT id, status, provider FROM generation_jobs ORDER BY id DESC LIMIT 10;"
```

---

## Troubleshooting Development Issues

### `ModuleNotFoundError` for project packages

The project uses local imports (e.g. `from generation.manifest import ...`). Always run commands from the project root with the `.venv` activated:

```powershell
cd c:\Users\poter\Downloads\synthetic
.venv\Scripts\Activate.ps1
python backend/worker.py  # ✓ correct
cd backend && python worker.py  # ✗ wrong — breaks relative imports
```

### Worker doesn't process jobs during tests

Expected behaviour. The embedded worker is suppressed in test mode. Call the worker functions directly:

```python
from backend.worker import _process_job
_process_job(db, job)
```

### Kaggle HTTP 401 errors

You are using the wrong `kaggle` package version. The system Python may have `kaggle 2.x`. Ensure you are inside the project `.venv`:

```powershell
.venv\Scripts\python -c "import kaggle; print(kaggle.__version__)"
# Should print 1.7.x
```

### `cp1252` / charmap encoding errors

The Kaggle notebook output contains non-ASCII characters. This is handled in `kaggle_provider.py` by using `ensure_ascii=True` when serializing manifests. If you see this error elsewhere, check that you are opening files with `encoding="utf-8"` and not relying on the system default encoding.

### Storage directory not found

The `storage/` directory is created automatically at startup. If it is missing:

```powershell
mkdir storage\accepted
mkdir storage\rejected
mkdir storage\jobs
mkdir storage\exports
```

Or simply restart the application — the lifespan handler creates it.

### Hot reload (`--reload`) not picking up changes in `generation/` or `deformation/`

Uvicorn's default reload watches only the `app` package. Add extra watch dirs:

```powershell
uvicorn backend.app.main:app --reload --reload-dir backend --reload-dir generation --reload-dir deformation --reload-dir validation
```
