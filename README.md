# Synthetic APAC Patient Face Dataset Generator

A privacy-safe system that generates **synthetic APAC (Asia-Pacific) patient face datasets** for rhinoplasty AI-training. Real patient photos cannot be used due to privacy regulations — this system creates photorealistic synthetic faces using the **FLUX.2-klein-4B** diffusion model running on Kaggle's free GPU, then applies clinical nasal deformations to simulate rhinoplasty conditions.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Generation Providers](#generation-providers)
- [API Reference](#api-reference)
- [Background Worker](#background-worker)
- [Nasal Deformation Pipeline](#nasal-deformation-pipeline)
- [Validation Pipeline](#validation-pipeline)
- [Kaggle Integration](#kaggle-integration)
- [Kaggle Pipeline Test](#kaggle-pipeline-test)
- [Diagnostic Suite](#diagnostic-suite)
- [Storage Layout](#storage-layout)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What it produces per patient

For each synthetic patient the system generates:

| Output | Count | Description |
|--------|-------|-------------|
| **Viewpoint images** | 9 | Frontal + Left/Right 20°/40°/60° + Basal + Helicopter |
| **Morphology variants** | 4 | Saddle · Hump · Broad · Deviated nose |
| **Binary masks** | 9 | Per-viewpoint nose segmentation mask |
| **Export ZIP** | 1 | Packaged for downstream ML training |

### System architecture

```
User / Frontend UI
       │  REST API
       ▼
   FastAPI App  ──────────────────────────────────────────────────────┐
   (backend/app/main.py)                                              │
       │                                                              │
   Routers: datasets · generation · jobs · deformation               │
            kaggle-test · images · languages                          │
       │                                                              │
   SQLite / PostgreSQL  ◄──────────────────────────────────────────  │
       │                                                              │
   Background Worker (backend/worker.py)                             │
       │                                                              │
   ┌───┴──────────────────────────────────────────────────────────┐  │
   │  MOCK provider          KAGGLE_BATCH provider                 │  │
   │  (local PNG, fast)      (FLUX.2-klein-4B on Kaggle GPU)      │  │
   │                              │                                │  │
   │                         1. Upload manifest JSON               │  │
   │                         2. Push .ipynb notebook               │  │
   │                         3. Poll kernel status                 │  │
   │                         4. Download artifacts                 │  │
   └───────────────────────────────────────────────────────────────┘  │
                                                                       │
   Validation Pipeline ──────────────────────────────────────────────┘
   Nasal Deformation Engine (BiSeNet → OpenCV warp)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Windows (primary platform; Linux/macOS compatible)
- Kaggle account with API token (for production generation)

### 1. Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

### 2. Configure environment

Copy and edit `backend/.env`:

```env
# Required
DATABASE_URL=sqlite:///./storage/apac_face.db
GENERATION_PROVIDER=MOCK          # or KAGGLE_BATCH for production

# Kaggle (required when GENERATION_PROVIDER=KAGGLE_BATCH)
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KERNEL_SLUG=your-username/apac-face-generator
KAGGLE_MANIFEST_DATASET_SLUG=your-username/apac-manifest
```

### 3. Start the application

```powershell
# Using the provided start script (recommended — uses project .venv)
start_app.cmd

# Or manually
.venv\Scripts\uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the UI

| URL | Description |
|-----|-------------|
| `http://localhost:8000/ui` | Main dataset manager |
| `http://localhost:8000/kaggle-test` | Kaggle pipeline test UI |
| `http://localhost:8000/docs` | Swagger / OpenAPI docs |
| `http://localhost:8000/health` | Health check |

---

## Project Structure

```
synthetic/
├── backend/
│   ├── .env                        # Environment configuration
│   ├── requirements.txt            # Python dependencies
│   ├── worker.py                   # Background job processor
│   └── app/
│       ├── main.py                 # FastAPI entry point
│       ├── core/
│       │   ├── config.py           # Settings (pydantic-settings)
│       │   ├── status.py           # State-machine enums
│       │   └── exceptions.py       # HTTP exception classes
│       ├── database/               # SQLAlchemy engine + session
│       ├── models/                 # ORM models (Dataset, GenerationJob, GeneratedImage)
│       ├── schemas/                # Pydantic request/response schemas
│       ├── services/               # Business logic layer
│       └── api/routes/             # FastAPI routers
│
├── generation/
│   ├── interface.py                # FaceGenerator ABC + ProviderOperationResult
│   ├── manifest.py                 # GenerationManifest model + prompt builder
│   ├── kaggle_provider.py          # KaggleBatchProvider (prepare/submit/poll/retrieve)
│   ├── mock_generator.py           # MockFaceGenerator (local colored PNG)
│   ├── provider_factory.py         # get_provider() factory
│   └── viewpoints.py              # 9 clinical viewpoint definitions
│
├── deformation/
│   ├── engine.py                   # Routes morphology × viewpoint → algorithm
│   ├── algorithms.py               # OpenCV warp algorithms
│   ├── masking.py                  # Mask load/save/clean utilities
│   ├── bisenet_inference.py        # BiSeNet ResNet-18 segmentor
│   └── reference_masks/           # Fallback binary masks per viewpoint
│
├── validation/
│   └── pipeline.py                 # validate_image() + baseline checks
│
├── diagnostic_suite.py             # Offline 7-category diagnostic runner
├── kaggle_notebook_check/
│   ├── apac_face_generator.ipynb  # Kaggle notebook (FLUX.2-klein-4B)
│   └── kernel-metadata.json       # Kaggle kernel push config
│
├── frontend/                       # Static HTML/JS UI
├── storage/                        # Runtime file storage (git-ignored)
└── start_app.cmd                   # Windows start script
```

---

## Configuration

All settings are loaded from `backend/.env` via **pydantic-settings**.

### Complete `.env` reference

```env
# ── Application ─────────────────────────────────────────────────────────────
APP_NAME=Synthetic APAC Patient Face Dataset Generator
APP_VERSION=1.0.0
ENVIRONMENT=development

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL=sqlite:///./storage/apac_face.db
# PostgreSQL: DATABASE_URL=postgresql://postgres:<password>@localhost:5432/apac_face_db

# ── Storage directories ───────────────────────────────────────────────────────
STORAGE_DIR=storage
GENERATED_IMAGES_DIR=storage/generated
ARTIFACT_DOWNLOAD_DIR=storage/artifacts_tmp
VALIDATED_IMAGES_DIR=storage/validated
REJECTED_IMAGES_DIR=storage/rejected
EXPORT_DIR=storage/exports

# ── Generation provider ───────────────────────────────────────────────────────
GENERATION_PROVIDER=MOCK           # MOCK | KAGGLE_BATCH
IMAGE_WIDTH=512
IMAGE_HEIGHT=512
MAX_REQUESTED_IMAGES=10000
ALLOWED_IMAGE_FORMATS=png,jpg,jpeg,webp
MAX_IMAGE_FILE_BYTES=20971520      # 20 MB
BATCH_SIZE=50

# ── Background worker ─────────────────────────────────────────────────────────
RUN_EMBEDDED_WORKER=true
WORKER_POLL_INTERVAL_SECONDS=10
JOB_LEASE_SECONDS=300              # 5 minutes
JOB_MAX_RETRIES=3

# ── Kaggle integration ────────────────────────────────────────────────────────
KAGGLE_USERNAME=your-username
# KAGGLE_KEY=your-32-char-api-key  # or use ~/.kaggle/kaggle.json
KAGGLE_KERNEL_SLUG=your-username/apac-face-generator
KAGGLE_MANIFEST_DATASET_SLUG=your-username/apac-manifest
KAGGLE_NOTEBOOK_PATH=kaggle_notebook_check/apac_face_generator.ipynb
KAGGLE_KERNEL_METADATA_PATH=kaggle_notebook_check/kernel-metadata.json
KAGGLE_POLL_INTERVAL_SECONDS=30
KAGGLE_RUN_TIMEOUT_SECONDS=7200   # 2 hours
KAGGLE_MAX_POLL_ERRORS=5
KAGGLE_POST_PUSH_DELAY_SECONDS=90  # wait after push before first poll

# ── Validation ────────────────────────────────────────────────────────────────
REQUIRE_FACE_DETECTION=false
ENABLE_DUPLICATE_DETECTION=true
DUPLICATE_HASH_THRESHOLD=10

# ── Security ─────────────────────────────────────────────────────────────────
JWT_SECRET=change_this_in_production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60
```

---

## Generation Providers

### MOCK (default, local)

- Generates a simple colored PNG with Pillow — no GPU required
- Deterministic: same seed → same image
- Used for development, CI, and testing
- Each viewpoint gets a distinct colored rectangle labeled with its name

### KAGGLE_BATCH (production)

Submits a Kaggle notebook that runs **FLUX.2-klein-4B** on a free P100 GPU.

**Prerequisites:**
1. Kaggle account with API token at `~/.kaggle/kaggle.json`
2. Create a **private Kaggle dataset** for the manifest: `your-username/apac-manifest`
3. Create (or fork) the **generation kernel**: `your-username/apac-face-generator`
4. Configure `KAGGLE_KERNEL_SLUG` and `KAGGLE_MANIFEST_DATASET_SLUG` in `.env`

**Lifecycle per job:**

```
QUEUED → PREPARING → SUBMITTED → RUNNING → DOWNLOADING → VALIDATING → PACKAGING → COMPLETED
                           ↑ post-push 90s delay ↑
```

> **Important:** Always start the app via `start_app.cmd` so the project `.venv`'s `kaggle 1.7.x` is used, not any system-installed `kaggle 2.x`. Mixing versions causes authentication failures.

---

## API Reference

Base URL: `http://localhost:8000`

### Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/datasets/` | Create a new dataset configuration |
| `GET` | `/datasets/` | List all datasets (paginated) |
| `GET` | `/datasets/{id}` | Get dataset details |
| `DELETE` | `/datasets/{id}` | Delete dataset (only if not generating) |
| `GET` | `/dashboard/stats` | Aggregate counts across all datasets |

**Create dataset — request body:**
```json
{
  "dataset_name": "South Asian Female Rhinoplasty Set 1",
  "region": "India",
  "population_group": "South Asian",
  "age_min": 25,
  "age_max": 35,
  "sex_or_attribute": "female",
  "skin_tone": "medium_brown",
  "pose": "frontal",
  "expression": "neutral",
  "num_images": 9,
  "resolution": 512,
  "medical_condition": "saddle_nose",
  "medical_severity": "moderate",
  "patient_id": "P01",
  "morphology_type": "saddle",
  "deformation_strength": 0.65,
  "deviated_direction": "right"
}
```

---

### Generation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/datasets/{id}/generate` | Enqueue a generation job (HTTP 202) |
| `GET` | `/datasets/{id}/generation-status` | Simplified status (legacy) |

**Generate response (HTTP 202):**
```json
{
  "job_id": 7,
  "dataset_id": 3,
  "status": "QUEUED",
  "provider": "KAGGLE_BATCH",
  "status_url": "/generation-jobs/7",
  "message": "Generation job 7 queued. Poll /generation-jobs/7 for progress."
}
```

---

### Generation Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/generation-jobs/{job_id}` | Full status, counters, timing |
| `GET` | `/generation-jobs/{job_id}/images` | Images produced by this job |
| `POST` | `/generation-jobs/{job_id}/cancel` | Request cancellation |
| `POST` | `/generation-jobs/{job_id}/retry` | Retry FAILED or COMPLETED_WITH_ERRORS |

**Job status response:**
```json
{
  "job_id": 7,
  "status": "RUNNING",
  "current_stage": "Generating viewpoint 5/9 (Right 20°)",
  "requested_count": 9,
  "accepted_count": 4,
  "progress_percentage": 44.4,
  "provider": "KAGGLE_BATCH",
  "provider_run_id": "username/apac-face-generator",
  "status_message": "Running image generation",
  "estimated_remaining_seconds": 120,
  "timing": { ... }
}
```

**Job status values:**

| Status | Meaning |
|--------|---------|
| `QUEUED` | Waiting for worker |
| `PREPARING` | Building manifest, uploading to Kaggle |
| `SUBMITTED` | Notebook pushed, waiting for Kaggle queue |
| `RUNNING` | Notebook executing on GPU |
| `DOWNLOADING` | Downloading artifacts from Kaggle |
| `VALIDATING` | Running image validation checks |
| `PACKAGING` | Finalizing + applying deformations |
| `COMPLETED` | All images accepted ✓ |
| `COMPLETED_WITH_ERRORS` | Some images rejected |
| `FAILED` | Job failed (see `error_summary`) |
| `CANCELLED` | Cancelled by user |

---

### Nasal Deformation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/deformation/types` | List available morphology types |
| `GET` | `/datasets/{id}/deformation/status` | Per-viewpoint deformation status |
| `POST` | `/datasets/{id}/deformation/single` | Apply to one viewpoint |
| `POST` | `/datasets/{id}/deformation/batch` | Apply to all 9 viewpoints |
| `GET` | `/datasets/{id}/deformation/image/{image_id}` | Serve deformed image |
| `GET` | `/datasets/{id}/deformation/mask/{image_id}` | Serve binary mask |

**Available morphology types:**

| Key | Name | Strength Range | Supports Direction |
|-----|------|---------------|-------------------|
| `saddle` | Saddle Nose | 0.1 – 1.0 | ✗ |
| `hump` | Hump Nose | 0.1 – 0.8 | ✗ |
| `broad` | Broad Nose | 0.1 – 0.8 | ✗ |
| `deviated` | Deviated Nose | 0.1 – 1.0 | ✓ (`left`/`right`) |

**Batch deformation request:**
```json
{
  "deformation_type": "saddle",
  "strength": 0.65,
  "direction": null,
  "patient_id": "P01"
}
```

---

### Images

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/datasets/{id}/images` | List all generated images |
| `GET` | `/images/{image_id}/file` | Serve original image file |

---

### Kaggle Pipeline Test

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/kaggle-test/prompt-preview` | Preview canonical FLUX prompt |
| `POST` | `/kaggle-test/run` | Start 10-stage end-to-end test |
| `GET` | `/kaggle-test/status/{run_id}` | Poll all 10 stage states |
| `GET` | `/kaggle-test/image/{run_id}` | Retrieve generated preview PNG |
| `POST` | `/kaggle-test/cleanup/{run_id}` | Clear in-memory image bytes |

---

### Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/datasets/{id}/export` | Build a ZIP export |
| `GET` | `/datasets/{id}/export` | Get export status / download |

---

## Background Worker

The worker is embedded inside the FastAPI process by default (`RUN_EMBEDDED_WORKER=true`). It can also run standalone:

```powershell
python -m backend.worker
```

### How it works

Every `WORKER_POLL_INTERVAL_SECONDS` (default 10 s):

1. **Stale job recovery** — Jobs whose lease expired are re-queued or failed
2. **Claim next QUEUED job** — Sets `worker_id` and `lease_expires_at`
3. **Process one step** per poll cycle:
   - **MOCK**: generates all 9 viewpoints synchronously, then applies deformation
   - **KAGGLE_BATCH**: advances one state per cycle (prepare → poll → poll → download → validate → package)

### Reliability features

| Feature | Mechanism |
|---------|-----------|
| Stale-job recovery | `lease_expires_at` checked each cycle; expired jobs re-queued |
| Retry limit | `JOB_MAX_RETRIES=3`; exceeded → permanent FAILED |
| Cancellation | `CANCEL_REQUESTED` checked before each image; honours mid-loop |
| Post-push delay | 90s wait before first Kaggle poll (timestamp in `current_stage`) |
| Poll error tracking | `POLL_ERRORS=N\|msg` prefix; fails at `KAGGLE_MAX_POLL_ERRORS=5` |
| Overlap guard | `batch_summary.json`'s `job_id` validated against current job |
| Graceful shutdown | SIGINT/SIGTERM → finishes current step before exiting |

---

## Nasal Deformation Pipeline

### 9 clinical viewpoints

| Key | Name | Angle | Camera position |
|-----|------|-------|----------------|
| `frontal` | Front 0° | 0° | Eye-level, straight |
| `left_20` | Left 20° | 20° | Slight oblique left |
| `left_40` | Left 40° | 40° | Three-quarter left |
| `left_60` | Left 60° | 60° | Strong lateral left |
| `right_20` | Right 20° | 20° | Slight oblique right |
| `right_40` | Right 40° | 40° | Three-quarter right |
| `right_60` | Right 60° | 60° | Strong lateral right |
| `basal` | Basal | — | Worm's-eye below nose |
| `helicopter` | Helicopter | — | Elevated overhead |

### Mask resolution (4-tier cascade)

```
Priority 1: BiSeNet ResNet-18 inference
            → Patient-specific deep semantic segmentation
            
Priority 2: Pre-stored patient mask
            → nasal-deformation-pipeline-main/data/identities/{patient_id}/masks/
            
Priority 3: Reference mask fallback
            → deformation/reference_masks/{viewpoint}/nose_binary.png
            
Priority 4: Geometric ellipse fallback
            → Anatomically-proportioned ellipse per viewpoint using cv2.ellipse()
```

### Deformation validation

After every deformation, pixel-level metrics are computed:
- `max_pixel_diff`, `mean_pixel_diff`, `changed_pixels_count`, `changed_pixels_pct`
- `mean_diff_inside_mask`

Result is `COMPLETED` or `NO_VISIBLE_CHANGE` (if `< 10 pixels changed` or `max_diff == 0`).

---

## Validation Pipeline

Every downloaded image passes through 7 baseline checks in order:

| # | Check | Failure cause |
|---|-------|--------------|
| 1 | `file_exists` | File not found (fatal) |
| 2 | `file_size` | Empty or exceeds 20 MB |
| 3 | `image_decodable` | Corrupt bytes, not a valid image (fatal) |
| 4 | `image_format` | Format not in `png,jpg,jpeg,webp` |
| 5 | `image_dimensions` | Width × height ≠ expected `512×512` |
| 6 | `check_checksum` | SHA-256 computed (always passes; populates context) |
| 7 | `check_duplicate` | SHA-256 matches a previously-seen image |
| 8 | `face_detection` | No face detected (optional; off by default) |

**Result:** ANY failed check → `REJECTED`. All passed → `ACCEPTED`. All skipped → `NOT_EVALUATED`.

---

## Kaggle Integration

### One-time setup

1. **Get Kaggle API credentials**
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → **Create New Token**
   - Save `kaggle.json` to `~/.kaggle/kaggle.json`
   - Or set `KAGGLE_USERNAME` + `KAGGLE_KEY` in `backend/.env`

2. **Create the manifest dataset** (private)
   - Kaggle → New Dataset → Name it `apac-manifest` → keep it private
   - Upload any placeholder file; the backend will create new versions automatically
   - Set `KAGGLE_MANIFEST_DATASET_SLUG=your-username/apac-manifest` in `.env`

3. **Create / fork the generation kernel**
   - Use or fork `kaggle_notebook_check/apac_face_generator.ipynb`
   - Kernel slug: `your-username/apac-face-generator`
   - Set `KAGGLE_KERNEL_SLUG=your-username/apac-face-generator` in `.env`

### How manifest delivery works

```
Backend                          Kaggle
  │                                │
  ├─ manifest.json ─────────────► Private dataset version (apac-manifest)
  │                                │
  ├─ notebook + metadata ────────► Kernel push (apac-face-generator)
  │                                │
  │                   notebook reads: /kaggle/input/apac-manifest/manifest.json
  │                   notebook writes: /kaggle/working/batch_summary.json + images/
  │                                │
  ◄──────────── download artifacts ─┘
```

### Notebook output contract

The notebook must write `/kaggle/working/batch_summary.json`:

```json
{
  "job_id": 7,
  "batch_id": "P01_abc123",
  "requested_count": 9,
  "successful_count": 9,
  "failed_count": 0,
  "images": [
    {
      "filename": "frontal.png",
      "seed": 123456,
      "prompt": "...",
      "width": 512,
      "height": 512,
      "viewtype": "frontal",
      "view_name": "Front (0°)",
      "patient_id": "P01"
    }
  ]
}
```

---

## Kaggle Pipeline Test

The `/kaggle-test` interface runs a **10-stage end-to-end test** of the full Kaggle path without touching any database.

### 10 stages

| Stage | Action | Type |
|-------|--------|------|
| 1 | Build canonical prompt | Local |
| 2 | Construct + validate `GenerationManifest` | Local |
| 3 | Upload manifest to Kaggle dataset | **Remote** |
| 4 | Submit notebook kernel | **Remote** |
| 5 | Poll kernel: queued → running | **Remote** |
| 6 | Confirm notebook completed | **Remote** |
| 7 | Download artifact ZIP | **Remote** |
| 8 | Locate image + run validation pipeline | Local |
| 9 | Cache image bytes for browser preview | Local |
| 10 | Delete temp workspace | Local |

### UI

Access at `http://localhost:8000/kaggle-test`:
- Configure patient demographics and prompt mode (canonical / custom)
- Click **Start Pipeline Test** to launch
- Watch stages update in real time
- Preview the generated image in-browser

### Safety

- **Single-flight** — only one test run at a time (HTTP 409 if another is active)
- **Zero database** — no Dataset / Job rows created
- **Temp isolation** — all files in `storage/temp_tests/{run_id}/`; deleted in Stage 10

---

## Diagnostic Suite

Runs 30+ local/mocked checks across 7 categories without touching Kaggle or the database.

```powershell
# Standard terminal report
python diagnostic_suite.py

# Machine-readable JSON
python diagnostic_suite.py --json

# Preview optional remote checks (dry run)
python diagnostic_suite.py --remote-check-dry-run
```

### Categories

| Cat | Focus | # Checks |
|-----|-------|---------|
| A | Configuration & local environment | 10 |
| B | Manifest preparation (mocked SDK) | 5 |
| C | Notebook submission (mocked SDK) | 4 |
| D | Polling & status mapping | 2 |
| E | Artifact retrieval & validation | 3 |
| F | API & browser UI endpoints | 3+ |
| G | Data isolation & cleanup | 2+ |

**Safety guarantees:**
- No remote mutations (no notebook push, no dataset version, no GPU cost)
- No secrets leakage (all outputs redacted)
- No database mutations
- MOCK safeguard: first check asserts `effective_provider == "MOCK"`

---

## Storage Layout

```
storage/
├── apac_face.db                    # SQLite database (dev)
├── datasets/
│   └── dataset_{N}/
│       └── P01/
│           ├── original/           # Generated viewpoint PNGs
│           │   ├── frontal.png
│           │   ├── left_20.png
│           │   ├── left_40.png
│           │   ├── left_60.png
│           │   ├── right_20.png
│           │   ├── right_40.png
│           │   ├── right_60.png
│           │   ├── basal.png
│           │   └── helicopter.png
│           ├── deformed/           # Legacy path
│           └── masks/              # Legacy path
├── deformations/
│   └── {dataset_id}/
│       └── P01/
│           ├── saddle/             # frontal_saddle.png, frontal_mask.png, …
│           ├── hump/
│           ├── broad/
│           └── deviated/
├── artifacts_tmp/
│   └── job_{N}/                    # Kaggle download staging (auto-cleaned)
├── generated/                      # Flat accepted images (legacy)
├── rejected/                       # Validation-rejected images
├── exports/                        # ZIP exports
└── temp_tests/                     # Kaggle test pipeline workspaces (auto-cleaned)
```

---

## Running Tests

```powershell
# Full test suite
cd c:\Users\poter\Downloads\synthetic
.venv\Scripts\python -m pytest backend/tests/ -v

# Diagnostic suite (offline, no Kaggle)
python diagnostic_suite.py

# Specific test file
.venv\Scripts\python -m pytest backend/tests/test_diagnostic_suite.py -v
```

---

## Troubleshooting

### `HTTP 401 Unauthorized` from Kaggle

The app is likely using system `kaggle 2.x` instead of project `.venv`'s `kaggle 1.7.x`.

**Fix:** Always start via `start_app.cmd` or explicitly:
```powershell
.venv\Scripts\uvicorn backend.app.main:app --reload
```

### Worker not processing jobs

Check embedded worker is enabled:
```env
RUN_EMBEDDED_WORKER=true
```

Or start standalone:
```powershell
.venv\Scripts\python -m backend.worker
```

### Kaggle kernel shows "complete" but no images downloaded

The post-push delay may be too short. Increase it:
```env
KAGGLE_POST_PUSH_DELAY_SECONDS=120
```

### `UnicodeDecodeError` during notebook push on Windows

The `_utf8_open_context()` patch should handle this. If you see it, verify you're running from the project `.venv` and not system Python.

### Deformation produces `NO_VISIBLE_CHANGE`

The nose mask may be too small or placed incorrectly. Check:
1. BiSeNet inference is working: `python -c "from deformation.bisenet_inference import bisenet_segmentor; print(bisenet_segmentor.is_available)"`
2. Reference masks exist in `deformation/reference_masks/{viewpoint}/nose_binary.png`
3. Increase `deformation_strength` (up to 1.0 for saddle, 0.8 for others)

### Job stuck in `SUBMITTED` for more than 10 minutes

Kaggle can queue notebooks. The `KAGGLE_POST_PUSH_DELAY_SECONDS=90` delay is intentional. The worker will begin polling after 90 seconds and report `RUNNING` once Kaggle starts execution.

---

## Security Notes

- **Credentials** — Never stored in source code. Loaded from `~/.kaggle/kaggle.json` or env vars only.
- **Error messages** — All 32+ character hex strings and `key=...` patterns are redacted from logs and API responses before exposure.
- **Manifest** — Contains no secrets. API keys, DB URLs, and JWT tokens are never included.
- **JWT secret** — Change `JWT_SECRET` in production. The default is a placeholder.
- **Path traversal** — Run IDs are validated against `^test_[a-fA-F0-9]{8,16}$` before any file access.

---

## License

For research and educational use. Synthetic faces generated by this system are entirely fictional and do not correspond to real persons.
