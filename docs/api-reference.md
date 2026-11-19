# API Reference

Complete REST API documentation for the Synthetic APAC Patient Face Dataset Generator.

**Base URL:** `http://localhost:8000`  
**OpenAPI / Swagger:** `http://localhost:8000/docs`  
**ReDoc:** `http://localhost:8000/redoc`

---

## Health

### `GET /`
Returns basic application metadata.

**Response:**
```json
{
  "message": "Synthetic APAC Patient Face Dataset Generator API",
  "version": "1.0.0",
  "status": "running",
  "provider": "KAGGLE_BATCH"
}
```

### `GET /health`
Full health check including Kaggle configuration.

**Response:**
```json
{
  "status": "healthy",
  "provider": "KAGGLE_BATCH",
  "kaggle_kernel": "username/apac-face-generator",
  "kaggle_manifest": "username/apac-manifest"
}
```

### `GET /dashboard/stats`
Aggregate statistics across all datasets.

---

## Datasets

### `POST /datasets/`
Create a new dataset configuration.

**Request body:**
```json
{
  "dataset_name": "South Asian Female Set 1",
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
  "custom_prompt": null,
  "viewpoint_prompts": null,
  "morphology_type": "saddle",
  "deformation_strength": 0.65,
  "deviated_direction": "right"
}
```

**Required fields:** `dataset_name`, `region`, `population_group`, `age_min`, `age_max`, `sex_or_attribute`, `skin_tone`, `pose`, `expression`, `num_images`, `resolution`

**Response `200`:**
```json
{
  "id": 3,
  "dataset_name": "South Asian Female Set 1",
  "status": "CREATED",
  "generated_count": 0,
  "created_at": "2026-11-19T07:00:00Z",
  ...
}
```

---

### `GET /datasets/`
List all datasets.

**Query parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip` | int | 0 | Records to skip |
| `limit` | int | 100 | Max records (1–500) |

---

### `GET /datasets/{id}`
Get a single dataset by ID.

**Response `404`:** Dataset not found.

---

### `DELETE /datasets/{id}`
Delete a dataset. Forbidden if the dataset has an active generation job.

**Response `409`:** Dataset is currently generating.  
**Response `404`:** Dataset not found.

---

### `POST /datasets/{id}/export`
Trigger a ZIP export of all accepted images.

**Request body:**
```json
{ "include_metadata": true }
```

---

### `GET /datasets/{id}/export`
Get export status or download the ZIP.

---

## Generation

### `POST /datasets/{id}/generate`
Enqueue an image generation job. Returns HTTP **202 Accepted**.

**Guards:**
- `409` if another active job exists for this dataset
- `409` if dataset is already `COMPLETED` or `COMPLETED_WITH_ERRORS`
- `404` if dataset not found

**Response `202`:**
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

### `GET /datasets/{id}/generation-status`
Simplified generation status (legacy endpoint).

**Response:**
```json
{
  "dataset_id": 3,
  "status": "GENERATING",
  "requested_images": 9,
  "generated_count": 4,
  "generation_attempts": 1,
  "progress_percentage": 44.44
}
```

---

## Generation Jobs

### `GET /generation-jobs/{job_id}`
Full job status with progress counters, timing, and stage information.

**Response:**
```json
{
  "job_id": 7,
  "dataset_id": 3,
  "provider": "KAGGLE_BATCH",
  "status": "RUNNING",
  "current_stage": "Kaggle notebook is running",
  "requested_count": 9,
  "generated_count": 0,
  "downloaded_count": 0,
  "validated_count": 0,
  "accepted_count": 0,
  "rejected_count": 0,
  "failed_count": 0,
  "progress_percentage": 15.0,
  "remaining_count": 9,
  "error_summary": null,
  "retry_count": 0,
  "provider_run_id": "username/apac-face-generator",
  "created_at": "2026-11-19T07:00:00",
  "started_at": "2026-11-19T07:00:05",
  "updated_at": "2026-11-19T07:02:30",
  "finished_at": null,
  "status_message": "Running image generation",
  "estimated_remaining_seconds": 1800,
  "timing": {
    "queue_waiting": {"duration_sec": 5.1, "status": "COMPLETED"},
    "manifest_preparation": {"duration_sec": 12.3, "status": "COMPLETED"},
    "kaggle_submission": {"duration_sec": 8.7, "status": "COMPLETED"},
    "flux_generation": {"started_at": "...", "status": "RUNNING"}
  }
}
```

**Job status values:**

| Value | Description | Terminal? |
|-------|-------------|-----------|
| `QUEUED` | Waiting for worker | ✗ |
| `PREPARING` | Building manifest, uploading | ✗ |
| `SUBMITTED` | Notebook pushed, waiting for Kaggle | ✗ |
| `RUNNING` | Notebook executing on GPU | ✗ |
| `CANCEL_REQUESTED` | Waiting for provider cancellation | ✗ |
| `DOWNLOADING` | Downloading artifacts | ✗ |
| `VALIDATING` | Running image validation | ✗ |
| `PACKAGING` | Finalizing + applying deformations | ✗ |
| `COMPLETED` | All images accepted ✓ | ✓ |
| `COMPLETED_WITH_ERRORS` | Some images rejected | ✓ |
| `FAILED` | Job failed permanently | ✓ |
| `CANCELLED` | Cancelled by user | ✓ |

---

### `GET /generation-jobs/{job_id}/images`
List images produced by this job.

**Response:**
```json
[
  {
    "id": 42,
    "dataset_id": 3,
    "job_id": 7,
    "image_url": "/images/42/file",
    "validation_status": "ACCEPTED",
    "generation_model": "FLUX.2-klein-4B",
    "seed": 123456,
    "created_at": "2026-11-19T08:00:00"
  }
]
```

---

### `POST /generation-jobs/{job_id}/cancel`
Request job cancellation.

- `QUEUED` jobs are cancelled immediately
- Active jobs move to `CANCEL_REQUESTED`; the worker will honour it
- `409` if the job is already terminal

---

### `POST /generation-jobs/{job_id}/retry`
Re-queue a `FAILED` or `COMPLETED_WITH_ERRORS` job.

- `409` if not retryable
- `429` if retry limit (`JOB_MAX_RETRIES=3`) exceeded

---

## Images

### `GET /datasets/{id}/images`
List all generated images for a dataset.

### `GET /images/{image_id}/file`
Serve the original generated image file (PNG).

---

## Nasal Morphology Deformation

### `GET /deformation/types`
List all available morphology types.

**Response:**
```json
{
  "morphologies": [
    {
      "type_key": "saddle",
      "name": "Saddle Nose",
      "description": "Depression or collapse of the mid-nasal bridge...",
      "default_strength": 0.65,
      "min_strength": 0.1,
      "max_strength": 1.0,
      "supports_direction": false,
      "default_direction": null,
      "clinical_notes": "Simulates supratip depression..."
    },
    {
      "type_key": "hump",
      "name": "Hump Nose",
      "default_strength": 0.40,
      "min_strength": 0.1,
      "max_strength": 0.8,
      "supports_direction": false
    },
    {
      "type_key": "broad",
      "name": "Broad Nose",
      "default_strength": 0.40,
      "supports_direction": false
    },
    {
      "type_key": "deviated",
      "name": "Deviated Nose",
      "default_strength": 0.5,
      "supports_direction": true,
      "default_direction": "right"
    }
  ]
}
```

---

### `GET /datasets/{id}/deformation/status`
Get per-viewpoint deformation status for a dataset.

**Query parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `patient_id` | string? | Override patient ID |

**Response:**
```json
{
  "dataset_id": 3,
  "patient_id": "P01",
  "status": "COMPLETED",
  "backend_integration_ready": true,
  "active_deformation_type": "saddle",
  "message": "All 9 viewpoints processed successfully (9 deformed).",
  "viewpoints": [
    {
      "viewpoint": "frontal",
      "viewpoint_name": "Front (0°)",
      "original_image_id": 42,
      "original_image_url": "/datasets/3/images/42",
      "deformation_status": "COMPLETED",
      "deformation_type": "saddle",
      "deformed_image_url": "/datasets/3/deformation/image/42",
      "mask_image_url": "/datasets/3/deformation/mask/42",
      "processing_notes": "[BiSeNet ResNet-18] Deformation validated: 1842px modified..."
    }
  ],
  "updated_at": "2026-11-19T08:30:00Z"
}
```

---

### `POST /datasets/{id}/deformation/single`
Apply deformation to one viewpoint.

**Request body:**
```json
{
  "deformation_type": "saddle",
  "strength": 0.65,
  "direction": null,
  "image_id": 42,
  "viewpoint": "frontal"
}
```

---

### `POST /datasets/{id}/deformation/batch`
Apply deformation to all 9 viewpoints at once.

**Request body:**
```json
{
  "deformation_type": "saddle",
  "strength": 0.65,
  "direction": null,
  "patient_id": "P01"
}
```

**Response:**
```json
{
  "dataset_id": 3,
  "patient_id": "P01",
  "deformation_type": "saddle",
  "strength": 0.65,
  "total_requested": 9,
  "succeeded_count": 9,
  "failed_count": 0,
  "status": "COMPLETED",
  "viewpoints": [...]
}
```

---

### `GET /datasets/{id}/deformation/image/{image_id}`
Serve the deformed image for an image record.

### `GET /datasets/{id}/deformation/mask/{image_id}`
Serve the binary nose segmentation mask.

---

## Kaggle Pipeline Test

### `POST /kaggle-test/prompt-preview`
Preview the canonical FLUX prompt without submitting anything.

**Request body:**
```json
{
  "population_group": "South Asian",
  "region": "India",
  "age_min": 25,
  "age_max": 35,
  "sex_or_attribute": "female",
  "skin_tone": "medium_brown",
  "pose": "frontal",
  "expression": "neutral",
  "lighting": "neutral"
}
```

**Response:**
```json
{
  "prompt": "Photorealistic synthetic portrait of a fictional person, South Asian, India region, age 25 to 35, female, skin tone medium_brown, frontal pose, neutral expression, neutral studio lighting. No glasses, no accessories, no heavy makeup, sharp facial details, plain neutral background, single person, centered face, realistic photography."
}
```

---

### `POST /kaggle-test/run`
Start a 10-stage end-to-end test run. Returns HTTP **202**.

**Request body:**
```json
{
  "population_group": "South Asian",
  "region": "India",
  "age_min": 25,
  "age_max": 35,
  "sex_or_attribute": "female",
  "skin_tone": "medium_brown",
  "pose": "frontal",
  "expression": "neutral",
  "lighting": "neutral",
  "resolution": 512,
  "prompt_mode": "canonical",
  "custom_prompt": null,
  "seed": null
}
```

**Returns `409`** if another test run is already in progress.

---

### `GET /kaggle-test/status/{run_id}`
Poll the stage-by-stage status of a test run.

**Response:**
```json
{
  "run_id": "test_a1b2c3d4",
  "status": "RUNNING",
  "current_stage_id": 5,
  "stages": [
    {"stage_id": 1, "name": "Prompt prepared", "status": "PASSED", "duration_seconds": 0.01},
    {"stage_id": 2, "name": "Manifest created and validated", "status": "PASSED", "duration_seconds": 0.05},
    {"stage_id": 3, "name": "Kaggle manifest dataset updated", "status": "PASSED", "duration_seconds": 8.3},
    {"stage_id": 4, "name": "Notebook submitted", "status": "PASSED", "duration_seconds": 5.1},
    {"stage_id": 5, "name": "Notebook running", "status": "RUNNING", "details": "Kernel is queued/running on Kaggle"},
    {"stage_id": 6, "name": "Notebook completed", "status": "PENDING"},
    ...
  ],
  "prompt": "Photorealistic synthetic portrait...",
  "seed": 987654321,
  "provider_run_id": "username/apac-face-generator",
  "image_url": null,
  "timing": {"total_elapsed_seconds": 240.5, "started_at": "..."}
}
```

---

### `GET /kaggle-test/image/{run_id}`
Retrieve the generated preview PNG from memory.

**Returns:** `image/png` binary  
**Headers:** `Cache-Control: no-store`

---

### `POST /kaggle-test/cleanup/{run_id}`
Clear in-memory image bytes for a completed run.

---

## Error responses

All errors follow:
```json
{ "error": "Human-readable message safe for display" }
```

| HTTP Code | When |
|-----------|------|
| `400` | Bad request (invalid parameters) |
| `404` | Resource not found |
| `409` | Conflict (duplicate job, already completed, can't cancel) |
| `429` | Too Many Requests (retry limit exceeded) |
| `500` | Internal server error |
