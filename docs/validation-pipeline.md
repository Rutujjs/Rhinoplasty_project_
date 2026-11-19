# Validation Pipeline

The validation pipeline (`validation/pipeline.py`) runs a sequence of checks on every downloaded image before it is accepted into the dataset.

---

## Table of Contents

- [Overview](#overview)
- [Outcomes](#outcomes)
- [Check Execution Order](#check-execution-order)
- [Baseline Checks Reference](#baseline-checks-reference)
- [Optional / Extension Checks](#optional--extension-checks)
- [ValidationContext Fields](#validationcontext-fields)
- [ValidationResult Fields](#validationresult-fields)
- [Failure Handling in the Worker](#failure-handling-in-the-worker)
- [Adding a Custom Check](#adding-a-custom-check)

---

## Overview

For each image, the pipeline:

1. Runs every registered check function in order.
2. Stops early on a **fatal failure** (`file_exists`, `image_decodable`).
3. Collects all check results and determines an aggregate outcome.
4. Returns a `ValidationResult`.

The worker uses this result to route each image to either `storage/accepted/` or `storage/rejected/` and writes a `GeneratedImage` DB record with the outcome.

---

## Outcomes

| Outcome | Condition |
|---|---|
| **ACCEPTED** | At least one check PASSED and no checks FAILED |
| **REJECTED** | At least one check FAILED |
| **NOT_EVALUATED** | All checks returned NOT_EVALUATED (no checks ran) |

A `NOT_EVALUATED` outcome on an individual check is **not a failure** — it means the check was skipped (e.g. a library was unavailable, or a feature was disabled in the context).

---

## Check Execution Order

The pipeline runs `BASELINE_CHECKS` in this exact order:

```
1. check_file_exists        [FATAL – stops pipeline on failure]
2. check_file_size
3. check_image_decodable    [FATAL – stops pipeline on failure]
4. check_image_format
5. check_image_dimensions
6. check_checksum
7. check_duplicate
8. check_face_detection     [optional]
```

> [!IMPORTANT]
> `check_file_exists` and `check_image_decodable` are **fatal checks**. If either fails, the pipeline immediately stops — no further checks are run. This avoids wasting time on subsequent checks for files that cannot be read.

---

## Baseline Checks Reference

### `check_file_exists`

**Purpose:** Verify the file exists on disk and is a regular file.

| Condition | Outcome |
|---|---|
| File not found | FAILED |
| Path is a directory | FAILED |
| File exists | PASSED |

**Fatal:** Yes — pipeline stops on failure.

---

### `check_file_size`

**Purpose:** Reject empty files and files exceeding the size limit.

| Condition | Outcome |
|---|---|
| Cannot stat file | FAILED |
| `size == 0` | FAILED — "File is empty" |
| `size > max_file_bytes` | FAILED — "File size N exceeds limit M" |
| Otherwise | PASSED |

**Side effect:** Sets `ctx.file_size_bytes`.

**Default limit:** 20 MB (`max_file_bytes = 20 * 1024 * 1024`).

---

### `check_image_decodable`

**Purpose:** Verify the file is a valid image that Pillow can open and fully decode.

| Condition | Outcome |
|---|---|
| Pillow not installed | NOT_EVALUATED |
| Cannot open/verify/load | FAILED — "Cannot decode image: …" |
| Successfully decoded | PASSED |

**Side effects:** Sets `ctx.image_width`, `ctx.image_height`, `ctx.image_format`.

**Fatal:** Yes — pipeline stops on failure.

---

### `check_image_format`

**Purpose:** Confirm the detected image format is in the allowed list.

| Condition | Outcome |
|---|---|
| Format not in `ctx.allowed_formats` | FAILED |
| Format allowed | PASSED |

**Default allowed formats:** `["png", "jpg", "jpeg", "webp"]`

The detected format comes from Pillow (`img.format.lower()`), not the file extension. If `ctx.image_format` is not yet set, the file extension is used as a fallback.

---

### `check_image_dimensions`

**Purpose:** Verify the image matches expected dimensions (if configured).

| Condition | Outcome |
|---|---|
| `expected_width` or `expected_height` is `None` | NOT_EVALUATED |
| `ctx.image_width` or `ctx.image_height` is `None` | NOT_EVALUATED |
| Width or height does not match | FAILED |
| Dimensions match | PASSED |

Configure expected dimensions in `ValidationContext`:

```python
ctx = ValidationContext(expected_width=1024, expected_height=1024)
```

If no expected dimensions are configured, this check always returns NOT_EVALUATED (non-blocking).

---

### `check_checksum`

**Purpose:** Compute the SHA-256 hash of the file and store it for downstream use.

| Condition | Outcome |
|---|---|
| Cannot read file | FAILED |
| Successfully hashed | PASSED |

**Side effect:** Sets `ctx.checksum_sha256`. This check **always passes** unless the file cannot be read.

> [!NOTE]
> This check runs before `check_duplicate` so the hash is available for the duplicate check.

---

### `check_duplicate`

**Purpose:** Detect exact duplicate images within the same batch using SHA-256.

| Condition | Outcome |
|---|---|
| `ctx.enable_duplicate_detection == False` | NOT_EVALUATED |
| `ctx.checksum_sha256` is `None` | NOT_EVALUATED |
| Hash is in `ctx.known_checksums` | FAILED — "Exact duplicate detected" |
| New hash | PASSED |

**Side effect:** Adds the hash to `ctx.known_checksums` on PASSED.

Duplicate detection is **per-pipeline-run** — `known_checksums` is not persisted across runs. To detect cross-batch duplicates, pre-populate `known_checksums` from previous runs.

Near-duplicate detection (perceptual hashing) is listed as a planned extension but is not currently implemented.

---

### `check_face_detection` (optional)

**Purpose:** Verify that at least one face is detectable in the image using OpenCV Haar cascades.

| Condition | Outcome |
|---|---|
| `ctx.require_face_detection == False` | NOT_EVALUATED |
| `opencv-python` not installed | NOT_EVALUATED |
| Cannot read image | FAILED |
| No face detected | FAILED — "No face detected" |
| Face found | PASSED |

**Default:** Disabled (`require_face_detection = False`). Enable by setting `ctx.require_face_detection = True`.

> [!TIP]
> Face detection is NOT_EVALUATED by default and does not block acceptance. It is only enforced if explicitly enabled in the `ValidationContext`.

---

## Optional / Extension Checks

Additional checks (age estimation, medical feature detection, privacy screening) are listed as future extension points. To register one:

```python
from validation.pipeline import BASELINE_CHECKS

def check_my_custom(path, ctx):
    # Return CheckResult with PASSED, FAILED, or NOT_EVALUATED
    ...

BASELINE_CHECKS.append(check_my_custom)
```

Or pass a custom list directly to `validate_image()`:

```python
result = validate_image(path, ctx, checks=[check_file_exists, check_my_custom])
```

---

## ValidationContext Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `expected_width` | `int \| None` | `None` | Required image width (pixels). `None` = not checked |
| `expected_height` | `int \| None` | `None` | Required image height (pixels). `None` = not checked |
| `allowed_formats` | `list[str]` | `["png","jpg","jpeg","webp"]` | Accepted image formats |
| `max_file_bytes` | `int` | `20971520` (20 MB) | Maximum allowed file size |
| `job_id` | `int \| None` | `None` | For logging/tracing only |
| `image_id_remote` | `str \| None` | `None` | Remote image ID for logging |
| `require_face_detection` | `bool` | `False` | Enforce face detection check |
| `enable_duplicate_detection` | `bool` | `True` | Enable exact-duplicate check |
| `known_checksums` | `set[str]` | `set()` | Pre-seeded checksums for duplicate detection |
| `duplicate_hash_threshold` | `int` | `10` | Reserved for near-duplicate threshold (not yet used) |

The following fields are **populated by the pipeline** during execution (do not set them manually):

| Field | Populated by |
|---|---|
| `image_width` | `check_image_decodable` |
| `image_height` | `check_image_decodable` |
| `image_format` | `check_image_decodable` |
| `file_size_bytes` | `check_file_size` |
| `checksum_sha256` | `check_checksum` |

---

## ValidationResult Fields

| Field | Type | Description |
|---|---|---|
| `path` | `Path` | Absolute path to the validated file |
| `outcome` | `str` | `"ACCEPTED"`, `"REJECTED"`, or `"NOT_EVALUATED"` |
| `check_results` | `list[CheckResult]` | All individual check outcomes |
| `rejection_reasons` | `list[str]` | Reasons from all FAILED checks |
| `image_width` | `int \| None` | Detected image width |
| `image_height` | `int \| None` | Detected image height |
| `image_format` | `str \| None` | Detected image format (lowercase) |
| `file_size_bytes` | `int \| None` | File size in bytes |
| `checksum_sha256` | `str \| None` | SHA-256 hex digest |

Properties:
- `result.accepted` → `True` if outcome is `"ACCEPTED"`
- `result.rejected` → `True` if outcome is `"REJECTED"`
- `result.rejection_reasons_json()` → JSON-encoded list of reason strings (stored in DB)

---

## Failure Handling in the Worker

After running the pipeline on all downloaded images, the worker:

1. Copies **accepted** images to `storage/accepted/`.
2. Copies **rejected** images to `storage/rejected/`.
3. Creates a `GeneratedImage` DB record for each image with `validation_status = ACCEPTED` or `REJECTED`.
4. If **all images are rejected** (zero accepted), transitions the job to `FAILED` with reason "All images failed validation".
5. If at least one image is accepted, proceeds to `PACKAGING`.

---

## Adding a Custom Check

```python
from pathlib import Path
from validation.pipeline import CheckResult, CheckOutcome, ValidationContext, BASELINE_CHECKS

def check_minimum_resolution(path: Path, ctx: ValidationContext) -> CheckResult:
    """Reject images smaller than 512×512."""
    if ctx.image_width is None or ctx.image_height is None:
        return CheckResult("minimum_resolution", CheckOutcome.NOT_EVALUATED,
                           "Dimensions not available")
    if ctx.image_width < 512 or ctx.image_height < 512:
        return CheckResult("minimum_resolution", CheckOutcome.FAILED,
                           f"Image {ctx.image_width}×{ctx.image_height} is below 512×512 minimum")
    return CheckResult("minimum_resolution", CheckOutcome.PASSED)

# Register globally
BASELINE_CHECKS.append(check_minimum_resolution)
```

Your check function signature must be:
```python
def check_<name>(path: Path, context: ValidationContext) -> CheckResult
```
