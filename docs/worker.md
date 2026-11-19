# Background Worker

The background worker (`backend/worker.py`) is the execution engine that drives every generation job through its full lifecycle — from queue to completed ZIP export.

---

## Table of Contents

- [Modes of Operation](#modes-of-operation)
- [Worker Identity & Leasing](#worker-identity--leasing)
- [Poll Loop](#poll-loop)
- [Job Lifecycle: MOCK Provider](#job-lifecycle-mock-provider)
- [Job Lifecycle: KAGGLE\_BATCH Provider](#job-lifecycle-kaggle_batch-provider)
- [Stale Job Recovery](#stale-job-recovery)
- [Concurrency & Overlap Guard](#concurrency--overlap-guard)
- [Cancellation Protocol](#cancellation-protocol)
- [Signals & Graceful Shutdown](#signals--graceful-shutdown)

---

## Modes of Operation

### Embedded Mode (default)

The worker runs as a **daemon thread** inside the FastAPI process. It starts automatically when the app starts (via the `lifespan` context in `main.py`) and stops when the app shuts down.

```python
# Called by FastAPI lifespan:
start_embedded_worker()   # starts thread
stop_embedded_worker()    # signals shutdown, joins thread (3 s timeout)
```

Embedded mode is **suppressed during testing**: if `PYTEST_CURRENT_TEST` is in the environment, or `pytest` is in `sys.modules`, or `RUN_EMBEDDED_WORKER=false` is set, the worker thread never starts.

### Standalone Mode

Run the worker as a separate process (e.g. for scaling or debugging):

```bash
cd c:\Users\poter\Downloads\synthetic
.venv\Scripts\python backend/worker.py
```

The process runs the same `run_once()` poll loop on a configurable interval (`WORKER_POLL_INTERVAL_SECONDS`, default 2 s) until a SIGINT/SIGTERM is received.

### `wake_worker()`

Any code path that creates a new job calls `wake_worker()`. This:
1. Auto-starts the embedded thread if it isn't running (and we're not in test mode).
2. Sets a `threading.Event` that immediately wakes the sleeping poll loop so the new job is picked up without waiting for the next poll interval.

---

## Worker Identity & Leasing

Each worker process generates a **unique worker ID** at startup:

```python
WORKER_ID = f"worker-{uuid4().hex[:8]}"
```

When a worker claims a job, it:
1. Sets `job.worker_id = WORKER_ID`
2. Sets `job.lease_expires_at = now + WORKER_LEASE_TTL_SECONDS` (default: 300 s)

The lease is **renewed** periodically during long-running Kaggle polling. If a worker crashes, the lease expires and another worker (or the same worker after restart) can adopt the orphaned job.

---

## Poll Loop

`run_once(db)` is called every poll cycle. It performs 4 steps in order:

```
1. Stale job recovery   → requeue any jobs whose lease has expired
2. QUEUED job dispatch  → claim up to 5 QUEUED jobs and process them
3. In-flight Kaggle     → advance each active KAGGLE_BATCH job one step
4. Fast-dispatch        → if a job completed this cycle, immediately pick up the next QUEUED job
```

The embedded worker runs `run_once()` on a tight loop:

```
sleep = min(WORKER_POLL_INTERVAL_SECONDS, 2.0)  # wake early if wake_worker() is called
```

---

## Job Lifecycle: MOCK Provider

MOCK jobs run entirely in-process, synchronously. The worker generates placeholder images for all 9 viewpoints, runs validation, then kicks off automatic nasal deformation.

```
QUEUED
  → PREPARING    (claim_job)
  → RUNNING      (generating images)
  → VALIDATING   (image pipeline)
  → PACKAGING    (deformation + ZIP)
  → COMPLETED
```

### Step-by-step

| Stage | What happens |
|---|---|
| **PREPARING** | Job claimed, dataset record confirmed |
| **RUNNING** | For each of 9 viewpoints: creates a solid-colour placeholder PNG (or uses a real face API if configured), writes it to `storage/jobs/<job_id>/images/<patient_id>/<viewpoint>/` |
| **VALIDATING** | Runs the full validation pipeline on each downloaded image. Accepts or rejects each one, writes DB records |
| **PACKAGING** | Calls `deformation_service.process_all_viewpoints()` for the dataset's morphology type, then `build_export_zip()` to produce the downloadable ZIP |
| **COMPLETED** | Dataset status set to `COMPLETED` |

---

## Job Lifecycle: KAGGLE\_BATCH Provider

Kaggle jobs span multiple poll cycles because Kaggle notebook execution takes minutes. The job is advanced one step per cycle; the worker never blocks waiting for Kaggle.

```
QUEUED
  → PREPARING    (claim_job, build manifest, push dataset version)
  → SUBMITTED    (kernel pushed / run triggered)
  → RUNNING      (notebook executing on Kaggle GPU)
  → DOWNLOADING  (notebook finished, downloading output)
  → VALIDATING   (image validation pipeline)
  → PACKAGING    (nasal deformation + ZIP)
  → COMPLETED
```

### Stage details

#### PREPARING → SUBMITTED (`_process_kaggle_job`)

1. Builds a `GenerationManifest` (patient demographics, viewpoints, model, seed).
2. Serializes manifest to `manifest.json` (using `ensure_ascii=True` to avoid Windows cp1252 crash).
3. Pushes a new version of the manifest Kaggle dataset (`apac-manifest`).
4. **Waits 90 seconds** after the push before starting to poll — this avoids reading the previous run's status.
5. Triggers the Kaggle kernel run (`kernel_push`).
6. Transitions job to `SUBMITTED`.

#### SUBMITTED / RUNNING (`_kaggle_poll`)

Called each poll cycle. Fetches kernel status from the Kaggle API:

| Kaggle status | Job status |
|---|---|
| `running` | `RUNNING` |
| `complete` | `DOWNLOADING` |
| `error` / `failed` | `FAILED` |
| `cancelled` | `CANCELLED` |
| (timeout) | `FAILED` after `KAGGLE_KERNEL_TIMEOUT_SECONDS` |

The lease is renewed on each poll cycle.

#### DOWNLOADING (`_kaggle_download`)

1. Calls `KaggleBatchProvider.download_results()`.
2. Downloads the kernel output ZIP from Kaggle.
3. Extracts images and `batch_summary.json` to `storage/jobs/<job_id>/`.
4. Validates that `batch_summary.json` contains the correct `job_id` to prevent cross-job contamination.
5. Transitions to `VALIDATING`.

#### VALIDATING (`_kaggle_validate`)

1. Reads each image from the downloaded output directory.
2. Runs the full **validation pipeline** on each image.
3. Copies accepted images to `storage/accepted/`, rejected to `storage/rejected/`.
4. Creates `GeneratedImage` DB records with validation status, dimensions, checksum.
5. If **zero images** are accepted → transitions to `FAILED`.
6. Otherwise → transitions to `PACKAGING`.

#### PACKAGING (`_kaggle_package`)

1. Reads `dataset.morphology_type` (e.g. `"saddle"`).
2. If morphology is not `"none"`:
   - Calls `deformation_service.process_all_viewpoints()` with the dataset's strength and direction settings.
3. Calls `build_export_zip()` to produce `storage/exports/<dataset_id>/<batch_id>.zip`.
4. Even if the ZIP fails, the job is marked `COMPLETED` (ZIP failure is non-fatal).
5. Dataset status set to `COMPLETED`.

---

## Stale Job Recovery

A job is considered **stale** when:
- It is in an active (non-terminal) state, AND
- `job.lease_expires_at < now`

`requeue_stale_job()` resets the job to `QUEUED`, clears the worker ID and lease, and logs a warning. The job will be re-claimed on the next poll cycle.

Default lease TTL: **300 seconds** (5 minutes). Long Kaggle polls renew the lease automatically.

---

## Concurrency & Overlap Guard

**Single-flight Kaggle execution:** Only one `KAGGLE_BATCH` job may be in an active state at a time. Before claiming or advancing a Kaggle job, the worker counts all other active Kaggle jobs. If `active_count > 0`, the job is skipped for this cycle.

This guarantees the manifest dataset always contains exactly one job's manifest, so the notebook cannot mix up outputs from concurrent jobs.

**MOCK jobs** have no overlap guard — multiple MOCK jobs can run concurrently.

---

## Cancellation Protocol

A job can be cancelled via `POST /api/v1/jobs/{id}/cancel`. This sets `job.status = CANCEL_REQUESTED`.

On the next poll:
- **MOCK jobs**: The worker checks `_shutdown_requested` between viewpoints and stops early.
- **Kaggle jobs**: `_kaggle_poll()` calls `kernel_cancel()` on the Kaggle API, then transitions to `CANCELLED`.

---

## Signals & Graceful Shutdown

The standalone worker registers `SIGINT` and `SIGTERM` handlers:

```python
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)
```

The handler sets `_shutdown_requested = True`. The poll loop checks this flag between jobs and exits cleanly after finishing the current job.

The embedded worker uses `stop_embedded_worker()`:
1. Sets `_embedded_shutdown` event.
2. Sets `_wake_event` (so the loop wakes immediately).
3. Joins the thread with a 3-second timeout.
