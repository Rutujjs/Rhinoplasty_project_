# Diagnostic Suite

`diagnostic_suite.py` is a comprehensive audit tool that verifies the entire Kaggle pipeline end-to-end across 7 dimensions — without touching Kaggle, your database, or any remote resource.

---

## Table of Contents

- [Safety Guarantees](#safety-guarantees)
- [Running the Suite](#running-the-suite)
- [Check Categories](#check-categories)
- [Interpreting Results](#interpreting-results)
- [JSON Output Mode](#json-output-mode)
- [What to Do When Checks Fail](#what-to-do-when-checks-fail)

---

## Safety Guarantees

The diagnostic suite is **completely safe to run at any time**:

1. **Zero Remote Mutations** — Never pushes a notebook, creates a dataset version, or submits a GPU job.
2. **Zero Secrets Leakage** — Redacts all tokens, passwords, and private values from outputs and JSON summaries.
3. **Zero Database Mutations** — Never inserts or modifies any database rows or counters.
4. **MOCK Safeguard Enforcement** — Verifies that your local `GENERATION_PROVIDER` setting is `MOCK` before any check runs.

---

## Running the Suite

> [!IMPORTANT]
> Always use the project `.venv` Python to run the suite. The system Python may have incompatible library versions.

```powershell
# Activate the virtual environment
cd c:\Users\poter\Downloads\synthetic
.venv\Scripts\Activate.ps1

# Standard human-readable terminal report
python diagnostic_suite.py

# Machine-readable JSON output
python diagnostic_suite.py --json

# Preview optional read-only remote checks (dry-run)
python diagnostic_suite.py --remote-check-dry-run
```

---

## Check Categories

The suite audits **7 categories** (labelled A–G):

### [A] Configuration and Local Environment

Verifies that the application is correctly configured to run locally:

| Check | What it tests |
|---|---|
| Settings load | `Settings` can be instantiated without errors |
| `GENERATION_PROVIDER` | Set to `MOCK` for local runs |
| Required env vars | `DATABASE_URL`, `STORAGE_PATH`, etc. are set |
| `BASE_DIR` resolution | Project root resolves correctly |
| `.env` file presence | `.env` file exists in the project root |
| venv Python | Running inside the project `.venv` |
| Package imports | Key packages (`fastapi`, `sqlalchemy`, `kaggle`, `PIL`, `cv2`) are importable |

---

### [B] Manifest Preparation

Verifies that the manifest system can build and serialize generation manifests:

| Check | What it tests |
|---|---|
| `GenerationManifest` creation | Manifest object can be constructed with valid fields |
| `build_prompt_from_attributes()` | Prompt builder returns a non-empty string |
| `ManifestProviderConfig` | Provider config serializes correctly |
| JSON serialization | Manifest serializes to valid JSON |
| `ensure_ascii=True` | JSON output contains no non-ASCII characters (Windows cp1252 safety) |
| Manifest path resolution | `_resolve_kernel_metadata_path()` and `_resolve_notebook_path()` return existing paths |

---

### [C] Notebook Submission

Verifies the Kaggle submission path without making any real API calls (all Kaggle API calls are **mocked**):

| Check | What it tests |
|---|---|
| `_get_kaggle_api()` | Kaggle API client can be constructed |
| `KaggleBatchProvider` instantiation | Provider initializes without errors |
| Manifest dataset push (mocked) | `upload_manifest_dataset()` logic completes with mocked responses |
| Kernel push (mocked) | `kernel_push()` logic completes with mocked responses |
| `_sanitize_error_detail()` | Error messages are safely sanitized |
| `_utf8_open_context()` | UTF-8 file open context manager works |

---

### [D] Notebook Execution and Polling

Verifies the polling state machine logic (mocked Kaggle status responses):

| Check | What it tests |
|---|---|
| Status mapping: `running` | Maps to `JobStatus.RUNNING` |
| Status mapping: `complete` | Maps to `JobStatus.DOWNLOADING` |
| Status mapping: `error` | Maps to `JobStatus.FAILED` |
| Status mapping: `cancelled` | Maps to `JobStatus.CANCELLED` |
| Timeout detection | Jobs exceeding `KAGGLE_KERNEL_TIMEOUT_SECONDS` transition to FAILED |
| Lease renewal | Lease is extended during active polling |

---

### [E] Artifact Retrieval and Validation

Verifies the download, extract, and validate pipeline using synthetic test images:

| Check | What it tests |
|---|---|
| `batch_summary.json` parsing | Output manifest schema validates correctly |
| Job ID guard | Mismatch between `batch_summary.json` job_id and current job is detected |
| Image extraction | ZIP extraction into the job directory works |
| Validation pipeline | `validate_image()` accepts a valid synthetic PNG |
| Rejection on bad format | Pipeline rejects a non-image file |
| Rejection on duplicate | Pipeline rejects an exact duplicate (same SHA-256) |
| Accepted/rejected routing | Files are copied to correct directories |

---

### [F] API and Browser UI

Verifies that the FastAPI application starts correctly and all routes are registered:

| Check | What it tests |
|---|---|
| FastAPI app creation | `app` object is an `FastAPI` instance |
| Route registration | All expected routers (`/health`, `/datasets`, `/generation`, `/jobs`, `/images`, `/deformation`, `/kaggle-test`) are registered |
| OpenAPI schema | `app.openapi()` returns a valid schema dict |
| `KagglePipelineTestRunner` | Test runner can be instantiated |
| `MockTestProvider` | Mock provider can generate fake results |

---

### [G] Data Isolation and Resource Cleanup

Verifies that isolation mechanisms work correctly:

| Check | What it tests |
|---|---|
| Job ID guard enforcement | `batch_summary.json` with wrong job_id is rejected |
| Temp directory cleanup | Temporary extraction directories are removed after use |
| Secrets not in manifests | Serialized manifest JSON contains no API keys or tokens |
| `_safe_redact()` | Redaction function strips known secret patterns |
| Provider setting enforcement | MOCK setting prevents Kaggle API initialization in test paths |

---

## Interpreting Results

### Terminal Output

Each check is printed as one line:

```
[PASS]     A.01  Settings load                               12.3 ms
[PASS]     A.02  GENERATION_PROVIDER is MOCK                  0.1 ms
[FAIL]     A.03  kaggle package importable                    5.2 ms
           → ImportError: No module named 'kaggle'
           → Fix: pip install -r requirements.txt (in .venv)
[NOT RUN]  D.04  Timeout detection                            0.0 ms
```

### Summary Block

```
══════════════════════════════════════════════
 DIAGNOSTIC SUMMARY
──────────────────────────────────────────────
 Total checks:  42
 PASS:          38  (local: 35, remote: 3)
 FAIL:           3  (local: 3,  remote: 0)
 NOT RUN:        1
 Duration:       4.2 s
══════════════════════════════════════════════
```

### Check types

| Type | Meaning |
|---|---|
| `LOCAL` | Runs entirely on your machine |
| `MOCKED` | Makes real code calls but intercepts all external I/O |
| `REMOTE` | Makes read-only calls to external services (only with `--remote-check-dry-run`) |

---

## JSON Output Mode

```bash
python diagnostic_suite.py --json
```

Outputs a `DiagnosticReport` as JSON:

```json
{
  "timestamp": "2025-11-19T07:00:00Z",
  "effective_provider": "MOCK",
  "total_checks": 42,
  "passed_count": 38,
  "failed_count": 3,
  "not_run_count": 1,
  "duration_seconds": 4.2,
  "checks": [
    {
      "check_id": "A.01",
      "name": "Settings load",
      "category": "A",
      "check_type": "LOCAL",
      "status": "PASS",
      "duration_ms": 12.3,
      "error_summary": null,
      "remediation": null,
      "details": null
    }
  ]
}
```

> [!NOTE]
> All secrets are redacted from JSON output. The `_safe_redact()` function strips API keys, tokens, and passwords matching known patterns.

---

## What to Do When Checks Fail

| Failing check | Most likely cause | Fix |
|---|---|---|
| **A.01 Settings load** | Syntax error in `.env`, missing required var | Check `.env` for typos; compare with [README.md](../README.md#environment-configuration) |
| **A.02 GENERATION_PROVIDER** | Set to `KAGGLE_BATCH` | Change to `MOCK` for local testing; switch back for production |
| **A.xx package importable** | Package not installed in `.venv` | Run `.venv\Scripts\pip install -r requirements.txt` |
| **B.xx Manifest** | Schema mismatch in `GenerationManifest` | Check `generation/manifest.py` for validation errors |
| **C.xx KaggleBatchProvider** | Kaggle credentials missing/invalid | Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env` |
| **D.xx Polling** | State machine regression | Check `backend/app/core/status.py` for status enum changes |
| **E.xx Validation** | Pillow or OpenCV not installed | Run `pip install pillow opencv-python` in `.venv` |
| **E.xx Job ID guard** | Logic error in download handler | Review `_kaggle_download()` in `backend/worker.py` |
| **F.xx Route registration** | Router not mounted in `main.py` | Check `app.include_router()` calls in `backend/app/main.py` |
| **G.xx Secrets in manifest** | Credentials being serialized | Review `GenerationManifest` serialization; ensure no settings objects are embedded |
