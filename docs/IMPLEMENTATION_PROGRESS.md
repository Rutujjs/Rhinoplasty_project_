# Synthetic APAC Patient Face Dataset Generator – Implementation Progress

> **Last updated:** 2026-10-14
> **Update this file after every significant implementation step.**

---

## Architecture Overview

```
synthetic/
├── backend/          FastAPI app + background worker
│   ├── app/
│   │   ├── api/routes/    datasets.py  generation.py  images.py  jobs.py
│   │   ├── core/          config.py  exceptions.py  status.py
│   │   ├── database/      base.py  connection.py  session.py
│   │   ├── models/        dataset.py  generated_image.py  generation_job.py
│   │   ├── schemas/       dataset.py  generation.py  job.py
│   │   └── services/      dataset_service.py  generation_service.py  image_service.py  job_service.py
│   └── worker.py          Background job-processing loop
├── generation/       Provider abstraction layer
│   ├── interface.py       FaceGenerator ABC + ProviderOperationResult
│   ├── manifest.py        GenerationManifest (Pydantic)
│   ├── mock_generator.py  MockFaceGenerator (synchronous, PIL)
│   └── provider_factory.py  get_provider() + registry
├── validation/
│   └── pipeline.py        validate_image() + BASELINE_CHECKS
├── storage/
│   └── generated/         Mock PNG output directory
└── frontend/         [EMPTY – not yet started]
```

---

## Phase Completion Status

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | DB models, dataset CRUD endpoints, mock generator | COMPLETE |
| **2** | Async job queue, worker loop, manifest, provider interface | COMPLETE |
| **3** | Job endpoints (GET/cancel/retry), generation-status (backward-compat) | COMPLETE |
| **4** | Validation pipeline skeleton, image metadata columns | COMPLETE |
| **5** | Fix test suite to match async API contract | COMPLETE |
| **5a** | KaggleBatchProvider implementation | COMPLETE |
| **5b** | Worker validation integration (full _kaggle_validate) | COMPLETE |
| **6** | ZIP export implementation | COMPLETE |
| **7** | Frontend (React/HTML dashboard) | COMPLETE |
| **8** | Kaggle provider real API calls + artifact import + tests | COMPLETE (mocked; real run not yet performed) |

---

## Current Task

All planned phases (1–8) are COMPLETE. 80/80 tests pass.

### Phase 8 – Kaggle integration (completed 2026-10-14)
- Implemented all 5 `KaggleBatchProvider` methods with real `kaggle` SDK calls.
- Added overlap guard in `run_once`: only one KAGGLE_BATCH job per kernel slug may be active.
- Added post-push delay (KAGGLE_POST_PUSH_DELAY_SECONDS) to avoid stale status reads.
- Added consecutive transient-error counter (POLL_ERRORS in error_summary) with configurable max.
- Replaced `_kaggle_validate` passthrough with full validation + DB import loop.
- Artifact overlap guard: batch_summary.json job_id is verified against the current job.
- Added `KAGGLE_MANIFEST_DATASET_SLUG` and `KAGGLE_POST_PUSH_DELAY_SECONDS` to config.
- Added `kaggle==1.7.4.5` to requirements.txt.
- Added `backend/tests/test_kaggle_provider.py` (33 tests: 20 unit + 13 worker integration).
- No real Kaggle run has been performed. All Kaggle API calls are mocked in tests.

---

## Remaining TODOs

| ID | Area | Description |
|----|------|-------------|
| I-4 | Kaggle | No real end-to-end Kaggle run has been performed |
| I-7 | Frontend | Pagination, filtering, image detail modal |

---

## Key Design Decisions

- Job queue is database-backed (no Redis/Celery needed).
- MOCK provider runs synchronously in the worker (all images in one pass).
- KAGGLE_BATCH provider uses a multi-step poll loop (PREPARING -> SUBMITTED -> RUNNING -> DOWNLOADING -> VALIDATING -> PACKAGING -> COMPLETED).
- SQLite in tests, PostgreSQL in production.
- generation_service.py still exists for legacy backward-compat (get_dataset_images / get_generated_image_by_id).

---

## Known Issues / TODOs

| ID | Area | Description |
|----|------|-------------|
| I-1 | Tests | ~~test_generation.py expects HTTP 200 + image list~~ – FIXED |
| I-2 | Generation | ~~generation/kaggle_provider.py missing~~ – FIXED (skeleton added) |
| I-3 | Tests | ~~No tests for /generation-jobs/ endpoints~~ – FIXED |
| I-4 | Worker | _kaggle_validate is a placeholder -- no real validation integration |
| I-5 | Export | ~~_kaggle_package is a placeholder~~ – FIXED (real ZIP export) |
| I-6 | Frontend | ~~Empty -- not started~~ – COMPLETE (frontend/index.html) |

---

## Running Tests

From project root, with venv activated (use `backend\venv_new`):
    python -m pytest backend/tests/ -v --tb=short

Current result: **47 passed** (test_datasets, test_generation, test_export, test_jobs).
