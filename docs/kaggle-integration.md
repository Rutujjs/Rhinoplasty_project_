# Kaggle Integration Guide

Complete instructions for configuring and operating the **KAGGLE_BATCH** generation provider.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Kaggle account | Free account at [kaggle.com](https://www.kaggle.com) |
| API credentials | Download from **Settings → API → Create New Token** |
| Private manifest dataset | Created once manually (instructions below) |
| Generation kernel | Fork or upload `apac_face_generator.ipynb` |
| Project `.venv` | Must use `kaggle 1.7.x` (NOT system `kaggle 2.x`) |

---

## Step 1 — Kaggle API credentials

Download `kaggle.json` from [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**.

Place it at `~/.kaggle/kaggle.json`:
```json
{"username": "your-username", "key": "your-32-char-key"}
```

Alternatively, set environment variables in `backend/.env`:
```env
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-32-char-api-key
```

> **Never** commit credentials to source control.

---

## Step 2 — Create the manifest dataset (once)

The backend delivers the generation manifest to the notebook by uploading it as a new **dataset version** on Kaggle.

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Name it `apac-manifest` (or any slug you choose)
3. Set visibility to **Private**
4. Upload any placeholder file (e.g., an empty `manifest.json`)
5. Note your dataset slug: `your-username/apac-manifest`

Set in `backend/.env`:
```env
KAGGLE_MANIFEST_DATASET_SLUG=your-username/apac-manifest
```

---

## Step 3 — Set up the generation kernel

The kernel is a Kaggle notebook (`apac_face_generator.ipynb`) that reads the manifest and runs FLUX.2-klein-4B.

### Option A — Use the provided notebook

1. Go to [kaggle.com/kernels](https://www.kaggle.com/kernels) → **New Notebook**
2. Upload `kaggle_notebook_check/apac_face_generator.ipynb`
3. Configure it:
   - **Enable GPU** (P100)
   - **Enable internet** access
   - **Add dataset**: attach your `apac-manifest` dataset as an input
4. Save the kernel; note its slug: `your-username/apac-face-generator`

### Option B — Fork an existing kernel

Fork the reference kernel and update `kernel-metadata.json`:
```json
{
  "id": "your-username/apac-face-generator",
  "title": "APAC Face Generator",
  "code_file": "apac_face_generator.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["your-username/apac-manifest"]
}
```

Set in `backend/.env`:
```env
KAGGLE_KERNEL_SLUG=your-username/apac-face-generator
KAGGLE_NOTEBOOK_PATH=kaggle_notebook_check/apac_face_generator.ipynb
KAGGLE_KERNEL_METADATA_PATH=kaggle_notebook_check/kernel-metadata.json
```

---

## Step 4 — Enable Kaggle provider

```env
GENERATION_PROVIDER=KAGGLE_BATCH
```

Always start the app via `start_app.cmd` or:
```powershell
.venv\Scripts\uvicorn backend.app.main:app --reload
```

> **Why?** The project `.venv` has `kaggle 1.7.x`. System Python may have `kaggle 2.x` which uses a different API and causes `HTTP 401 Unauthorized` on dataset uploads.

---

## How manifest delivery works

```
Backend (backend/worker.py)
  │
  ├─ 1. Write manifest.json + dataset-metadata.json to temp dir
  │
  ├─ 2. api.dataset_create_version(folder=tmp_dir) ──────────────► Kaggle
  │      Uploaded as new version of: your-username/apac-manifest        │
  │                                                                      │
  ├─ 3. api.kernels_push(tmp_dir) ───────────────────────────────► Kaggle Kernels
  │      Pushes: apac_face_generator.ipynb + kernel-metadata.json        │
  │                                                                      │
  │  Kaggle executes the notebook with:                                  │
  │    - Input: /kaggle/input/apac-manifest/manifest.json                │
  │    - Output: /kaggle/working/batch_summary.json + images/            │
  │                                                                      │
  └─ 4. api.kernels_output(slug, path=dest_dir) ◄────────────────── Kaggle ZIP
         Downloads and unpacks to: storage/artifacts_tmp/job_{N}/
```

---

## Polling behavior

After submitting, the worker waits **`KAGGLE_POST_PUSH_DELAY_SECONDS`** (default 90 s) before the first status poll. This avoids reading the previous run's status, which Kaggle's API may briefly return after a push.

**Kaggle status → internal outcome mapping:**

| Kaggle status | Outcome |
|--------------|---------|
| `queued`, `running`, `newscript`, `cancel_requested` | `PENDING` |
| `complete`, `completed` | `SUCCESS` |
| `error`, `failed`, `cancelled`, `cancel_acknowledged` | `PERMANENT_ERROR` |

**Polling loop:**
- Interval: `KAGGLE_POLL_INTERVAL_SECONDS` (default 30 s)
- Timeout: `KAGGLE_RUN_TIMEOUT_SECONDS` (default 7200 s / 2 hours)
- Max consecutive errors: `KAGGLE_MAX_POLL_ERRORS` (default 5)

---

## Notebook output contract

The notebook **must** write `/kaggle/working/batch_summary.json`:

```json
{
  "job_id": 7,
  "batch_id": "P01_abc12345",
  "requested_count": 9,
  "successful_count": 9,
  "failed_count": 0,
  "patient_id": "P01",
  "images": [
    {
      "filename": "frontal.png",
      "seed": 123456,
      "prompt": "Photorealistic synthetic portrait...",
      "width": 512,
      "height": 512,
      "viewtype": "frontal",
      "view_name": "Front (0°)",
      "patient_id": "P01"
    },
    {
      "filename": "left_20.png",
      "viewtype": "left_20",
      "view_name": "Left 20°",
      ...
    }
  ]
}
```

The notebook must also write images to `/kaggle/working/rhinoplasty_dataset/batches/{batch_id}/images/`.

---

## Troubleshooting Kaggle issues

### HTTP 401 Unauthorized

- Caused by running from system Python instead of `.venv`
- Fix: use `start_app.cmd` or `.venv\Scripts\uvicorn ...`

### HTTP 403 Forbidden

- The authenticated Kaggle account does not own the manifest dataset
- Ensure `KAGGLE_MANIFEST_DATASET_SLUG=your-username/...` matches your account

### HTTP 404 Not Found

- The manifest dataset does not exist on Kaggle
- Create it manually via the Kaggle web UI first

### Kernel stuck in `queued` for 60+ minutes

- Kaggle free GPU queue can be long; `KAGGLE_RUN_TIMEOUT_SECONDS=7200` allows up to 2 hours
- Check the kernel execution page on kaggle.com to see GPU availability

### `No image files found in artifact download directory`

- The notebook did not write images to `/kaggle/working/`
- Check the notebook execution logs on Kaggle for FLUX model loading errors
- GPU must be enabled in the kernel settings

### Previous run status returned after push

- Expected behavior — handled by `KAGGLE_POST_PUSH_DELAY_SECONDS=90`
- If still occurring, increase to 120
