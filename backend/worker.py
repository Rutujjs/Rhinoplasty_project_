"""
Background worker – processes queued generation jobs.

Run from the project root:
    python -m backend.worker

The worker polls the database for QUEUED jobs at a configurable interval,
claims each job with a lease, runs it through the generation pipeline, and
marks it complete.  It can run alongside the FastAPI process or in a
separate process.

Architecture
------------
- Job queue: database-backed (no Redis/Celery required).
- Stale-job recovery: jobs whose lease has expired are re-queued.
- Retry limits: a job is permanently failed after JOB_MAX_RETRIES retries.
- Each claimed job is processed synchronously within the worker's poll loop.
  For MOCK, this is fast.  For KAGGLE_BATCH, the worker submits, polls the
  remote status, and retrieves artifacts over multiple iterations.

Shutdown
--------
Send SIGINT (Ctrl-C) or SIGTERM to trigger a graceful shutdown.  The worker
finishes the current job step before exiting.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import socket
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on the path when run as  python -m backend.worker
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings
from backend.app.core.status import DatasetStatus, JobStatus
from backend.app.database.session import SessionLocal
from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage
from backend.app.models.generation_job import GenerationJob
from backend.app.services.job_service import (
    claim_job,
    get_queued_jobs,
    get_stale_jobs,
    renew_lease,
    requeue_stale_job,
    transition_job_status,
    update_progress,
)
from generation.provider_factory import get_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")

# --------------------------------------------------------------------------- #
# Shutdown handling                                                             #
# --------------------------------------------------------------------------- #

_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    logger.info("Shutdown signal received (%s). Finishing current step…", signum)
    _shutdown_requested = True


try:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
except (ValueError, AttributeError):
    pass

# --------------------------------------------------------------------------- #
# Worker identity                                                               #
# --------------------------------------------------------------------------- #

WORKER_ID = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# Main processing logic                                                         #
# --------------------------------------------------------------------------- #

def _determine_viewpoint(filename: str, meta: dict) -> tuple[str, str]:
    from generation.viewpoints import CLINICAL_VIEWPOINTS, VIEWPOINT_MAP
    # 1. Prioritize explicit viewpoint metadata
    vt = meta.get("viewtype") or meta.get("view_key")
    if vt:
        vt_clean = str(vt).lower().strip()
        if vt_clean in VIEWPOINT_MAP:
            return vt_clean, meta.get("view_name") or VIEWPOINT_MAP[vt_clean].name
    # 2. Fall back to filename heuristic
    fn_lower = filename.lower()
    for spec in CLINICAL_VIEWPOINTS:
        if spec.key in fn_lower:
            return spec.key, spec.name
    if "front" in fn_lower:
        return "frontal", "Front (0°)"
    return "frontal", "Front (0°)"


def _process_mock_job(db, job: GenerationJob, dataset: Dataset) -> None:
    """
    Process a MOCK provider job synchronously.

    Supports both full 9-viewpoint clinical patient datasets and arbitrary
    requested_count image requests with incremental progress tracking
    (~11% per viewpoint for 9 views, 100% on completion).
    """
    import os
    from random import randint
    from backend.app.schemas.generation import GenerationRequest
    from backend.app.core.status import ImageValidationStatus
    from datetime import datetime, timezone
    from generation.viewpoints import CLINICAL_VIEWPOINTS
    from generation.manifest import build_prompt_from_attributes

    provider = get_provider("MOCK")

    db.refresh(job)
    if job.status in (JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED):
        logger.info("Job %d cancelled before generation started", job.id)
        if job.status != JobStatus.CANCELLED:
            transition_job_status(db, job, JobStatus.CANCELLED, stage="Cancelled by user")
        dataset.status = DatasetStatus.CREATED
        db.commit()
        return

    # Start with RUNNING status and initial stage (10% progress)
    transition_job_status(
        db, job, JobStatus.RUNNING,
        stage="Starting generation",
    )
    update_progress(
        db, job,
        stage="Starting generation",
        progress_percentage=10.0,
        commit=True,
    )

    generated_images = []
    failed = 0
    patient_id = dataset.patient_id or f"P{dataset.id:02d}"

    # Prepare directories with strict dataset-level isolation
    from generation.mock_generator import OUTPUT_DIR
    dataset_storage_dir = Path(settings.STORAGE_DIR) / "datasets" / f"dataset_{dataset.id}" / patient_id
    (dataset_storage_dir / "original").mkdir(parents=True, exist_ok=True)
    (dataset_storage_dir / "deformed").mkdir(parents=True, exist_ok=True)
    (dataset_storage_dir / "masks").mkdir(parents=True, exist_ok=True)

    # Also keep patient_root for backward compatibility
    patient_root = OUTPUT_DIR / patient_id
    (patient_root / "original").mkdir(parents=True, exist_ok=True)
    (patient_root / "deformed").mkdir(parents=True, exist_ok=True)
    (patient_root / "masks").mkdir(parents=True, exist_ok=True)

    base_prompt = dataset.custom_prompt or build_prompt_from_attributes(
        population_group=dataset.population_group,
        region=dataset.region,
        age_min=dataset.age_min,
        age_max=dataset.age_max,
        sex_or_attribute=dataset.sex_or_attribute,
        skin_tone=dataset.skin_tone,
        pose=dataset.pose,
        expression=dataset.expression,
        medical_condition=dataset.medical_condition,
        medical_severity=dataset.medical_severity,
    )

    total_images = job.requested_count
    base_seed = randint(100_000, 800_000)
    is_testing = "PYTEST_CURRENT_TEST" in os.environ

    for idx in range(total_images):
        renew_lease(db, job)

        # Check for cancellation before starting viewpoint
        db.refresh(job)
        if job.status == JobStatus.CANCEL_REQUESTED:
            logger.info("Job %d: cancellation acknowledged during generation", job.id)
            transition_job_status(
                db, job, JobStatus.CANCELLED,
                stage="Cancelled by user",
            )
            dataset.status = DatasetStatus.CREATED if len(generated_images) == 0 else DatasetStatus.COMPLETED_WITH_ERRORS
            dataset.generated_count = len(generated_images)
            db.commit()
            return

        spec = CLINICAL_VIEWPOINTS[idx % len(CLINICAL_VIEWPOINTS)]
        seed = base_seed + (idx * 100)

        # Update stage before generating
        stage_desc = f"Generating viewpoint {idx + 1}/{total_images} ({spec.name})"
        job.current_stage = stage_desc
        db.commit()

        try:
            req = GenerationRequest(
                age_min=dataset.age_min,
                age_max=dataset.age_max,
                sex_or_attribute=dataset.sex_or_attribute,
                skin_tone=dataset.skin_tone,
                population_group=dataset.population_group,
                region=dataset.region,
                pose=dataset.pose,
                expression=dataset.expression,
                medical_condition=dataset.medical_condition,
                medical_severity=dataset.medical_severity,
                resolution=dataset.resolution,
                patient_id=patient_id,
                viewpoint=spec.key,
                custom_prompt=dataset.custom_prompt,
            )

            dest_file = dataset_storage_dir / "original" / f"{spec.key}.png"
            if hasattr(provider, "_draw_mock_image"):
                provider._draw_mock_image(req, seed, spec, patient_id, dest_file)
            else:
                res = provider.generate(req)
                dest_file = Path(res.image_path)

            # Also maintain copy in legacy patient_root for backward compatibility
            try:
                legacy_patient_file = patient_root / "original" / f"{spec.key}.png"
                if dest_file != legacy_patient_file:
                    shutil.copy2(dest_file, legacy_patient_file)
            except Exception:
                pass

            # Also create legacy file for legacy tests expecting flat mock_{seed}.png
            legacy_file = OUTPUT_DIR / f"mock_{seed}.png"
            try:
                legacy_file.write_bytes(dest_file.read_bytes())
            except Exception:
                pass

            now = datetime.now(timezone.utc).replace(tzinfo=None)
            prompt_str = f"{spec.name}: {base_prompt}" if total_images == 9 else base_prompt

            image_record = GeneratedImage(
                dataset_id=dataset.id,
                job_id=job.id,
                batch_id=job.batch_id,
                image_path=str(dest_file),
                generation_model="MOCK",
                seed=seed,
                prompt=prompt_str,
                status="GENERATED",
                validation_status=ImageValidationStatus.ACCEPTED,
                validated_at=now,
                patient_id=patient_id,
                viewpoint=spec.key,
                viewpoint_name=spec.name,
                deformation_status="NOT_STARTED",
            )
            db.add(image_record)
            db.flush()
            generated_images.append(image_record)

        except Exception as exc:
            logger.exception("Image %d/%d failed for job %d: %s", idx + 1, total_images, job.id, exc)
            failed += 1

        done_count = len(generated_images)
        pct = round((done_count / total_images) * 100.0, 1) if total_images > 0 else 100.0
        update_progress(
            db, job,
            generated=done_count,
            accepted=done_count,
            failed=failed,
            progress_percentage=pct,
            stage=f"Generated {done_count}/{total_images} viewpoints ({spec.name})" if failed == 0 else f"Generated {done_count}/{total_images} ({failed} failed)",
            commit=True,
        )

        # Small delay for UI realism when not running tests
        if not is_testing:
            time.sleep(0.08)

    # Update dataset counts.
    dataset.generated_count = len(generated_images)
    dataset.generation_attempts = (dataset.generation_attempts or 0) + 1
    db.commit()

    # Automatic Nasal Deformation: Apply user's selected morphology to all viewpoints
    morphology = getattr(dataset, "morphology_type", None) or "saddle"
    if morphology and morphology.lower() != "none" and len(generated_images) > 0:
        try:
            from backend.app.schemas.deformation import BatchDeformationRequest
            from backend.app.services.deformation_service import deformation_service

            job.current_stage = f"Applying nasal deformation ({morphology})"
            db.commit()

            batch_req = BatchDeformationRequest(
                deformation_type=morphology,
                strength=float(getattr(dataset, "deformation_strength", 0.6) or 0.6),
                direction=getattr(dataset, "deviated_direction", "right") or "right",
                patient_id=patient_id,
            )
            deformation_service.process_all_viewpoints(db, dataset.id, batch_req)
            logger.info("Job %d: automated nasal deformation completed for dataset %d", job.id, dataset.id)
        except Exception as def_err:
            logger.exception("Job %d: automated nasal deformation failed: %s", job.id, def_err)

    # Choose terminal status.
    if failed == 0 and len(generated_images) == job.requested_count:
        terminal_status = JobStatus.COMPLETED
        dataset.status = DatasetStatus.COMPLETED
    elif len(generated_images) > 0:
        terminal_status = JobStatus.COMPLETED_WITH_ERRORS
        dataset.status = DatasetStatus.COMPLETED_WITH_ERRORS
    else:
        terminal_status = JobStatus.FAILED
        dataset.status = DatasetStatus.FAILED

    db.commit()

    transition_job_status(
        db, job, terminal_status,
        stage=f"Completed: {len(generated_images)} accepted, {failed} failed",
    )
    if terminal_status == JobStatus.COMPLETED:
        job.progress_percentage = 100.0
        db.commit()

    logger.info(
        "Job %d completed: %d accepted, %d failed (status=%s)",
        job.id, len(generated_images), failed, terminal_status.value,
    )


def _process_kaggle_job(db, job: GenerationJob, dataset: Dataset) -> None:
    """
    Process a KAGGLE_BATCH provider job.

    This is a multi-step, multi-poll operation:
      PREPARING → SUBMITTED → RUNNING → DOWNLOADING → VALIDATING → PACKAGING → COMPLETED

    The worker completes one logical step per call and returns.
    On the next poll cycle it resumes from the current status.
    """
    current = JobStatus(job.status)

    if current == JobStatus.PREPARING:
        _kaggle_prepare(db, job, dataset)
    elif current == JobStatus.SUBMITTED:
        _kaggle_poll(db, job)
    elif current == JobStatus.RUNNING:
        _kaggle_poll(db, job)
    elif current == JobStatus.DOWNLOADING:
        _kaggle_download(db, job, dataset)
    elif current == JobStatus.VALIDATING:
        _kaggle_validate(db, job, dataset)
    elif current == JobStatus.PACKAGING:
        _kaggle_package(db, job, dataset)


def _kaggle_prepare(db, job: GenerationJob, dataset: Dataset) -> None:
    """Build and upload the generation manifest; then submit the kernel."""
    from generation.manifest import build_manifest
    from pathlib import Path

    logger.info("Job %d: preparing Kaggle batch", job.id)

    patient_id = dataset.patient_id or f"P{dataset.id:02d}"
    batch_id = job.batch_id or f"{patient_id}_{uuid.uuid4().hex[:8]}"
    manifest = build_manifest(
        dataset_id=dataset.id,
        job_id=job.id,
        dataset=dataset,
        requested_count=1,
        image_width=settings.IMAGE_WIDTH,
        image_height=settings.IMAGE_HEIGHT,
        provider_name="KAGGLE_BATCH",
        batch_id=batch_id,
    )

    # Save manifest locally for later use during validation.
    manifest_dir = Path(settings.ARTIFACT_DOWNLOAD_DIR) / f"job_{job.id}"
    manifest_path = manifest_dir / "manifest.json"
    manifest.save(manifest_path)

    job.batch_id = manifest.batch_id
    db.flush()

    provider = get_provider("KAGGLE_BATCH")
    result = provider.prepare_batch(manifest)

    from backend.app.services.timing_service import record_stage_end, record_stage_start

    if not result.succeeded:
        record_stage_end(job, "manifest_preparation", status="FAILED", error=result.message)
        transition_job_status(
            db, job, JobStatus.FAILED,
            stage="Kaggle preparation failed",
            error_summary=result.message,
        )
        return

    record_stage_end(job, "manifest_preparation", status="COMPLETED")
    record_stage_start(job, "kaggle_submission")

    # Submit.
    submit_result = provider.submit_batch(manifest)
    if not submit_result.succeeded:
        record_stage_end(job, "kaggle_submission", status="FAILED", error=submit_result.message)
        transition_job_status(
            db, job, JobStatus.FAILED,
            stage="Kaggle submission failed",
            error_summary=submit_result.message,
        )
        return

    record_stage_end(job, "kaggle_submission", status="COMPLETED")
    record_stage_start(job, "flux_generation")

    # provider_run_id = kernel slug (the only stable Kaggle run handle).
    job.provider_run_id = submit_result.provider_run_id

    # Record the push time so the poll loop can honour the post-push delay.
    # We store it in current_stage as a timestamp prefix; the worker reads
    # it back before making the first API status call.
    from datetime import datetime, timezone
    push_ts = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    transition_job_status(
        db, job, JobStatus.SUBMITTED,
        stage=f"PUSH_TS={push_ts}|Submitted to Kaggle (kernel={submit_result.provider_run_id})",
    )


def _kaggle_poll(db, job: GenerationJob) -> None:
    """Poll Kaggle for run status; advance to DOWNLOADING when complete.

    Respects the post-push delay: the first poll after submission waits
    KAGGLE_POST_PUSH_DELAY_SECONDS to avoid reading the previous run's status.
    Tracks consecutive poll errors and fails the job after KAGGLE_MAX_POLL_ERRORS.
    """
    from datetime import datetime, timezone

    # Enforce post-push delay on the very first poll after SUBMITTED.
    if JobStatus(job.status) == JobStatus.SUBMITTED:
        stage = job.current_stage or ""
        if stage.startswith("PUSH_TS="):
            try:
                ts_part = stage.split("|")[0].replace("PUSH_TS=", "")
                push_time = datetime.fromisoformat(ts_part)
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                elapsed = (now - push_time).total_seconds()
                delay = settings.KAGGLE_POST_PUSH_DELAY_SECONDS
                if elapsed < delay:
                    remaining = int(delay - elapsed)
                    logger.info(
                        "Job %d: post-push delay — waiting %ds more before first poll",
                        job.id,
                        remaining,
                    )
                    renew_lease(db, job)
                    return
            except (ValueError, IndexError) as exc:
                logger.warning("Job %d: could not parse PUSH_TS from stage: %s", job.id, exc)

    provider = get_provider("KAGGLE_BATCH")
    result = provider.poll_status(job.provider_run_id)

    if result.is_pending:
        # Check overall run timeout before renewing the lease.
        if job.started_at is not None:
            from datetime import datetime, timezone as _tz
            now = datetime.now(_tz.utc).replace(tzinfo=None)
            elapsed = (now - job.started_at).total_seconds()
            timeout = settings.KAGGLE_RUN_TIMEOUT_SECONDS
            if elapsed > timeout:
                logger.error(
                    "Job %d: Kaggle run timed out after %.0fs (limit=%ds)",
                    job.id,
                    elapsed,
                    timeout,
                )
                transition_job_status(
                    db, job, JobStatus.FAILED,
                    stage="Kaggle run timeout",
                    error_summary=(
                        f"Kaggle kernel did not complete within "
                        f"{timeout}s timeout"
                    ),
                )
                return
        renew_lease(db, job)
        if JobStatus(job.status) == JobStatus.SUBMITTED:
            transition_job_status(db, job, JobStatus.RUNNING, stage="Kaggle notebook is running")
        return

    if result.succeeded:
        from backend.app.services.timing_service import record_stage_end, record_stage_start
        record_stage_end(job, "flux_generation", status="COMPLETED")
        record_stage_start(job, "artifact_download")
        transition_job_status(db, job, JobStatus.DOWNLOADING, stage="Downloading artifacts")
        return

    if result.outcome.name == "TRANSIENT_ERROR":
        # Increment a transient-error counter stored in error_summary.
        current_errors = _parse_poll_error_count(job.error_summary)
        current_errors += 1
        logger.warning(
            "Job %d: transient Kaggle poll error %d/%d: %s",
            job.id,
            current_errors,
            settings.KAGGLE_MAX_POLL_ERRORS,
            result.message,
        )
        if current_errors >= settings.KAGGLE_MAX_POLL_ERRORS:
            transition_job_status(
                db, job, JobStatus.FAILED,
                stage="Too many consecutive poll errors",
                error_summary=(
                    f"Kaggle polling failed {current_errors} times: {result.message}"
                ),
            )
        else:
            job.error_summary = f"POLL_ERRORS={current_errors}|{result.message}"
            renew_lease(db, job)
        return

    # Permanent error (kernel error / cancelled).
    transition_job_status(
        db, job, JobStatus.FAILED,
        stage="Kaggle notebook ended with error",
        error_summary=result.message,
    )


def _parse_poll_error_count(error_summary: str | None) -> int:
    """Extract consecutive poll error count from the error_summary field."""
    if error_summary and error_summary.startswith("POLL_ERRORS="):
        try:
            return int(error_summary.split("|")[0].replace("POLL_ERRORS=", ""))
        except ValueError:
            pass
    return 0


def _kaggle_download(db, job: GenerationJob, dataset: Dataset) -> None:
    from generation.manifest import GenerationManifest
    from pathlib import Path

    provider = get_provider("KAGGLE_BATCH")
    dest_dir = str(Path(settings.ARTIFACT_DOWNLOAD_DIR) / f"job_{job.id}")

    manifest_path = Path(dest_dir) / "manifest.json"
    if manifest_path.exists():
        manifest = GenerationManifest.load(manifest_path)
    else:
        manifest = None

    result = provider.retrieve_artifacts(
        job.provider_run_id,
        dest_dir,
        manifest,
    )

    if result.succeeded:
        from backend.app.services.timing_service import record_stage_end, record_stage_start
        record_stage_end(job, "artifact_download", status="COMPLETED")
        record_stage_start(job, "image_validation")
        # Clear any transient error counter from the polling phase.
        if job.error_summary and job.error_summary.startswith("POLL_ERRORS="):
            job.error_summary = None
            db.flush()
        transition_job_status(db, job, JobStatus.VALIDATING, stage="Validating images")
    elif result.is_error:
        from backend.app.services.timing_service import record_stage_end
        record_stage_end(job, "artifact_download", status="FAILED", error=result.message)
        transition_job_status(
            db, job, JobStatus.FAILED,
            stage="Artifact retrieval failed",
            error_summary=result.message,
        )
    else:
        renew_lease(db, job)


def _kaggle_validate(db, job: GenerationJob, dataset: Dataset) -> None:
    """
    Import and validate downloaded artifacts.

    The notebook saves images under:
        rhinoplasty_dataset/batches/{batch_id}/images/
    and the summary at:
        rhinoplasty_dataset/batches/{batch_id}/batch_summary.json

    This function:
      1. Finds batch_summary.json anywhere in the artifact tree.
      2. Reads per-image metadata (seed, prompt, dimensions).
         Accepts both 'generated_count' and 'successful_count' field names.
         Validates job_id if present; falls back to local manifest if absent.
      3. Collects images recursively from the artifact directory.
      4. Runs the validation pipeline on each image.
      5. Copies accepted images to GENERATED_IMAGES_DIR, rejected to
         REJECTED_IMAGES_DIR.
      6. Creates GeneratedImage records in the database.
      7. Advances to PACKAGING or FAILED.
    """
    from backend.app.core.status import ImageValidationStatus
    from backend.app.services.job_service import update_progress
    from validation.pipeline import ValidationContext, validate_image
    from datetime import datetime, timezone
    import shutil
    import json

    dest_dir = Path(settings.ARTIFACT_DOWNLOAD_DIR) / f"job_{job.id}"
    generated_dir = Path(settings.GENERATED_IMAGES_DIR)
    rejected_dir = Path(settings.REJECTED_IMAGES_DIR)
    generated_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Find batch_summary.json recursively in the artifact tree            #
    # ------------------------------------------------------------------ #
    image_meta: dict[str, dict] = {}   # filename → {seed, prompt, width, height}

    summary_candidates = sorted(dest_dir.rglob("batch_summary.json"))
    summary_path = summary_candidates[0] if summary_candidates else None

    if summary_path and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            # Overlap guard: validate job_id if present in summary.
            # The updated notebook writes job_id; older runs may omit it.
            # Fall back to reading the locally saved manifest.json.
            summary_job_id = summary.get("job_id")
            if summary_job_id is None:
                local_manifest_path = dest_dir / "manifest.json"
                if local_manifest_path.exists():
                    try:
                        lm = json.loads(local_manifest_path.read_text(encoding="utf-8"))
                        summary_job_id = lm.get("job_id")
                    except Exception:
                        pass

            if summary_job_id is not None and int(summary_job_id) != job.id:
                logger.error(
                    "_kaggle_validate: job %d received summary for job %d — "
                    "possible artifact overlap; failing job.",
                    job.id,
                    summary_job_id,
                )
                transition_job_status(
                    db, job, JobStatus.FAILED,
                    stage="Artifact overlap detected",
                    error_summary=(
                        f"batch_summary.json belongs to job {summary_job_id}, "
                        f"not this job {job.id}"
                    ),
                )
                return

            for img_entry in summary.get("images", []):
                fname = img_entry.get("filename", "")
                if fname:
                    image_meta[fname] = {
                        "seed":        img_entry.get("seed", 0),
                        "prompt":      img_entry.get("prompt", ""),
                        "width":       img_entry.get("width"),
                        "height":      img_entry.get("height"),
                        "viewtype":    img_entry.get("viewtype") or img_entry.get("view_key"),
                        "view_name":   img_entry.get("view_name"),
                        "patient_id":  img_entry.get("patient_id") or summary.get("patient_id"),
                    }
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "_kaggle_validate: job %d batch_summary.json malformed (%s); "
                "proceeding without per-image metadata",
                job.id,
                exc,
            )

    # ------------------------------------------------------------------ #
    # Collect image files recursively                                      #
    # ------------------------------------------------------------------ #
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = sorted(
        f for f in dest_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions
    )

    if image_meta:
        # If batch_summary.json defined specific images (e.g. 9 canonical patient views),
        # prioritize those named files and exclude auxiliary duplicate batch artifacts.
        named_files = [f for f in image_files if f.name in image_meta]
        if named_files:
            image_files = named_files

    if not image_files:
        logger.error(
            "_kaggle_validate: job %d — no image files found under %s",
            job.id,
            dest_dir,
        )
        transition_job_status(
            db, job, JobStatus.FAILED,
            stage="No images found after download",
            error_summary="No image files found in artifact download directory",
        )
        return

    logger.info(
        "_kaggle_validate: job %d — validating %d image(s)",
        job.id,
        len(image_files),
    )

    # ------------------------------------------------------------------ #
    # Validate each image and register in DB                              #
    # ------------------------------------------------------------------ #
    accepted_count = 0
    rejected_count = 0
    known_checksums: set[str] = set()

    for img_path in image_files:
        meta = image_meta.get(img_path.name, {})

        view_key, view_name = _determine_viewpoint(img_path.name, meta)
        exp_w = meta.get("width")
        exp_h = meta.get("height")
        if not exp_w or not exp_h:
            if view_key == "frontal" and ("frontal" in img_path.name or "384" in img_path.name):
                exp_w = 384
                exp_h = 384
            else:
                exp_w = settings.IMAGE_WIDTH
                exp_h = settings.IMAGE_HEIGHT

        ctx = ValidationContext(
            expected_width=exp_w,
            expected_height=exp_h,
            allowed_formats=settings.allowed_formats,
            max_file_bytes=settings.MAX_IMAGE_FILE_BYTES,
            job_id=job.id,
            require_face_detection=settings.REQUIRE_FACE_DETECTION,
            enable_duplicate_detection=settings.ENABLE_DUPLICATE_DETECTION,
            known_checksums=known_checksums,
            duplicate_hash_threshold=settings.DUPLICATE_HASH_THRESHOLD,
        )

        result = validate_image(img_path, ctx)

        # Keep known_checksums updated for cross-image duplicate detection.
        if ctx.checksum_sha256:
            known_checksums.add(ctx.checksum_sha256)

        now = datetime.now(timezone.utc).replace(tzinfo=None)

        patient_id = meta.get("patient_id") or dataset.patient_id or f"P{dataset.id:02d}"

        # Multi-view patient storage layout with strict dataset-level isolation:
        # storage/datasets/dataset_{dataset_id}/{patient_id}/original/
        dataset_storage_dir = Path(settings.STORAGE_DIR) / "datasets" / f"dataset_{dataset.id}" / patient_id
        patient_orig_dir = dataset_storage_dir / "original"
        patient_orig_dir.mkdir(parents=True, exist_ok=True)
        (dataset_storage_dir / "deformed").mkdir(parents=True, exist_ok=True)
        (dataset_storage_dir / "masks").mkdir(parents=True, exist_ok=True)

        # Legacy patient folder for backward compatibility
        legacy_patient_dir = generated_dir / patient_id / "original"
        legacy_patient_dir.mkdir(parents=True, exist_ok=True)

        if result.accepted:
            # Save into dataset-isolated patient original folder
            mv_dest_path = patient_orig_dir / f"{view_key}.png"
            shutil.copy2(img_path, mv_dest_path)
            try:
                shutil.copy2(img_path, legacy_patient_dir / f"{view_key}.png")
            except Exception:
                pass
            # Also keep root copy for backward compatibility
            dest_path = generated_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            final_path = mv_dest_path
            v_status = ImageValidationStatus.ACCEPTED
            accepted_count += 1
        else:
            dest_path = rejected_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            final_path = dest_path
            v_status = ImageValidationStatus.REJECTED
            rejected_count += 1

        image_record = GeneratedImage(
            dataset_id=dataset.id,
            job_id=job.id,
            batch_id=job.batch_id,
            image_path=str(final_path),
            generation_model="FLUX.2-klein-4B",
            seed=meta.get("seed", 0),
            prompt=meta.get("prompt", ""),
            status="GENERATED",
            validation_status=v_status,
            rejection_reasons=(
                result.rejection_reasons_json() if result.rejected else None
            ),
            image_width=result.image_width,
            image_height=result.image_height,
            image_format=result.image_format,
            file_size_bytes=result.file_size_bytes,
            checksum_sha256=result.checksum_sha256,
            validated_at=now,
            patient_id=patient_id,
            viewpoint=view_key,
            viewpoint_name=view_name,
            deformation_status="NOT_STARTED",
        )
        db.add(image_record)
        db.flush()

    dataset.generated_count = accepted_count
    dataset.generation_attempts = (dataset.generation_attempts or 0) + 1

    update_progress(
        db, job,
        validated=accepted_count + rejected_count,
        accepted=accepted_count,
        rejected=rejected_count,
        stage=(
            f"Validated {accepted_count + rejected_count} images: "
            f"{accepted_count} accepted, {rejected_count} rejected"
        ),
        commit=False,
    )
    db.commit()

    logger.info(
        "_kaggle_validate: job %d — %d accepted, %d rejected",
        job.id,
        accepted_count,
        rejected_count,
    )

    if accepted_count == 0:
        transition_job_status(
            db, job, JobStatus.FAILED,
            stage="All images failed validation",
            error_summary=(
                f"0 of {accepted_count + rejected_count} images passed validation"
            ),
        )
        dataset.status = DatasetStatus.FAILED
        db.commit()
        return

    from backend.app.services.timing_service import record_stage_end, record_stage_start
    record_stage_end(job, "image_validation", status="COMPLETED")
    record_stage_start(job, "deformation")

    transition_job_status(
        db, job, JobStatus.PACKAGING,
        stage="Packaging dataset",
    )


def _kaggle_package(db, job: GenerationJob, dataset: Dataset) -> None:
    """Final packaging step – apply nasal deformation and build the ZIP export."""
    from backend.app.services.export_service import build_export_zip
    from backend.app.services.timing_service import record_stage_end, record_stage_start

    # Automatic Nasal Deformation: Apply user's selected morphology to all viewpoints
    morphology = getattr(dataset, "morphology_type", None) or "saddle"
    if morphology and morphology.lower() != "none":
        try:
            from backend.app.schemas.deformation import BatchDeformationRequest
            from backend.app.services.deformation_service import deformation_service

            job.current_stage = f"Applying nasal deformation ({morphology})"
            db.commit()

            patient_id = dataset.patient_id or f"P{dataset.id:02d}"
            batch_req = BatchDeformationRequest(
                deformation_type=morphology,
                strength=float(getattr(dataset, "deformation_strength", 0.6) or 0.6),
                direction=getattr(dataset, "deviated_direction", "right") or "right",
                patient_id=patient_id,
            )
            deformation_service.process_all_viewpoints(db, dataset.id, batch_req, job_id=job.id)
            record_stage_end(job, "deformation", status="COMPLETED")
            logger.info("Job %d: automated nasal deformation completed for dataset %d", job.id, dataset.id)
        except Exception as def_err:
            record_stage_end(job, "deformation", status="FAILED", error=str(def_err))
            logger.exception("Job %d: automated nasal deformation failed: %s", job.id, def_err)
    else:
        record_stage_end(job, "deformation", status="COMPLETED")

    record_stage_start(job, "persistence")
    logger.info("Job %d: packaging dataset %d into ZIP", job.id, dataset.id)
    try:
        zip_path = build_export_zip(db, dataset, include_rejected=False)
        logger.info("Job %d: ZIP export written to %s", job.id, zip_path)
    except Exception as exc:
        logger.warning(
            "Job %d: ZIP export failed (%s) – marking complete anyway", job.id, exc
        )

    record_stage_end(job, "persistence", status="COMPLETED")
    dataset.status = DatasetStatus.COMPLETED
    db.commit()
    transition_job_status(db, job, JobStatus.COMPLETED, stage="Completed")


def _process_job(db, job: GenerationJob) -> None:
    """Dispatch to the correct provider handler."""
    dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
    if dataset is None:
        logger.error("Job %d references missing dataset %d", job.id, job.dataset_id)
        transition_job_status(
            db, job, JobStatus.FAILED,
            stage="Dataset not found",
            error_summary="Dataset record not found in database",
        )
        return

    provider = job.provider.upper()
    try:
        if provider == "MOCK":
            # MOCK jobs are processed synchronously from QUEUED.
            # The claim_job call already moved status to PREPARING.
            # Patch the status back so _process_mock_job can transition to RUNNING.
            # We use a force-update here since PREPARING → RUNNING is valid.
            _process_mock_job(db, job, dataset)
        elif provider == "KAGGLE_BATCH":
            _process_kaggle_job(db, job, dataset)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    except Exception as exc:
        logger.exception("Unhandled error processing job %d", job.id)
        try:
            transition_job_status(
                db, job, JobStatus.FAILED,
                stage="Unhandled worker error",
                error_summary="An internal error occurred during generation",
            )
            dataset.status = DatasetStatus.FAILED
            dataset.generation_attempts = (dataset.generation_attempts or 0) + 1
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Failed to mark job %d as FAILED", job.id)


# --------------------------------------------------------------------------- #
# Poll loop                                                                     #
# --------------------------------------------------------------------------- #

def run_once(db) -> int:
    """
    Process one round of the poll loop.

    Returns the number of jobs processed in this round.
    """
    # 1. Recover stale jobs.
    stale = get_stale_jobs(db)
    for stale_job in stale:
        logger.warning("Recovering stale job %d", stale_job.id)
        requeue_stale_job(db, stale_job, worker_id=WORKER_ID)

    # 2. Claim and process queued jobs.
    queued = get_queued_jobs(db, limit=5)
    processed = 0

    for job in queued:
        if _shutdown_requested:
            break

        # Overlap guard for KAGGLE_BATCH: only one job per kernel slug may
        # be in an active (non-terminal) state at any time.  If another job
        # for the same kernel is already running, skip this job and try again
        # on the next poll cycle.
        if job.provider.upper() == "KAGGLE_BATCH":
            kernel_slug = settings.KAGGLE_KERNEL_SLUG
            if kernel_slug:
                active_count = (
                    db.query(GenerationJob)
                    .filter(
                        GenerationJob.provider == "KAGGLE_BATCH",
                        GenerationJob.status.in_([
                            JobStatus.PREPARING,
                            JobStatus.SUBMITTED,
                            JobStatus.RUNNING,
                            JobStatus.CANCEL_REQUESTED,
                            JobStatus.DOWNLOADING,
                            JobStatus.VALIDATING,
                            JobStatus.PACKAGING,
                        ]),
                        GenerationJob.id != job.id,
                    )
                    .count()
                )
                if active_count > 0:
                    logger.info(
                        "Job %d: skipping — another KAGGLE_BATCH job is already active",
                        job.id,
                    )
                    continue

        logger.info("Claiming job %d (provider=%s)", job.id, job.provider)
        try:
            claim_job(db, job, worker_id=WORKER_ID)
            _process_job(db, job)
            processed += 1
        except Exception:
            logger.exception("Failed to process job %d", job.id)

    # 3. Continue processing any KAGGLE_BATCH jobs in mid-flight states.
    # We query all active Kaggle jobs and adopt any orphaned jobs whose lease expired.
    active_kaggle = (
        db.query(GenerationJob)
        .filter(
            GenerationJob.provider == "KAGGLE_BATCH",
            GenerationJob.status.in_([
                JobStatus.SUBMITTED,
                JobStatus.RUNNING,
                JobStatus.DOWNLOADING,
                JobStatus.VALIDATING,
                JobStatus.PACKAGING,
                JobStatus.CANCEL_REQUESTED,
            ]),
        )
        .all()
    )
    job_completed_in_cycle = False
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for job in active_kaggle:
        if _shutdown_requested:
            break
        # If another worker holds an active (non-expired) lease on this job, skip it:
        if job.worker_id and job.worker_id != WORKER_ID:
            if job.lease_expires_at and job.lease_expires_at >= now:
                continue
            # Lease expired or orphaned -> adopt the job for this worker:
            logger.info(
                "Adopting orphaned in-flight Kaggle job %d (worker_id: %s → %s)",
                job.id,
                job.worker_id,
                WORKER_ID,
            )
            job.worker_id = WORKER_ID
            renew_lease(db, job)

        try:
            dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
            _kaggle_job_step(db, job, dataset)
            processed += 1
            if JobStatus(job.status).is_terminal:
                job_completed_in_cycle = True
        except Exception:
            logger.exception("Failed to advance Kaggle job %d", job.id)

    # 4. If an in-flight job reached a terminal state in this cycle,
    # immediately dispatch any pending QUEUED jobs without waiting for the next cycle.
    if job_completed_in_cycle:
        queued_next = get_queued_jobs(db, limit=5)
        for next_job in queued_next:
            if _shutdown_requested:
                break
            if next_job.provider.upper() == "KAGGLE_BATCH":
                active_count = (
                    db.query(GenerationJob)
                    .filter(
                        GenerationJob.provider == "KAGGLE_BATCH",
                        GenerationJob.status.in_([
                            JobStatus.PREPARING,
                            JobStatus.SUBMITTED,
                            JobStatus.RUNNING,
                            JobStatus.CANCEL_REQUESTED,
                            JobStatus.DOWNLOADING,
                            JobStatus.VALIDATING,
                            JobStatus.PACKAGING,
                        ]),
                        GenerationJob.id != next_job.id,
                    )
                    .count()
                )
                if active_count > 0:
                    continue

            logger.info("Immediately claiming next queued job %d after completion of previous job", next_job.id)
            try:
                claim_job(db, next_job, worker_id=WORKER_ID)
                _process_job(db, next_job)
                processed += 1
                break  # Claim only one Kaggle job to respect single-flight execution
            except Exception:
                logger.exception("Failed to dispatch next job %d", next_job.id)

    return processed


def _kaggle_job_step(db, job, dataset):
    """Advance a Kaggle job one step forward in its lifecycle."""
    current = JobStatus(job.status)
    if current in (JobStatus.SUBMITTED, JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED):
        _kaggle_poll(db, job)
    elif current == JobStatus.DOWNLOADING:
        _kaggle_download(db, job, dataset)
    elif current == JobStatus.VALIDATING:
        _kaggle_validate(db, job, dataset)
    elif current == JobStatus.PACKAGING:
        _kaggle_package(db, job, dataset)


# --------------------------------------------------------------------------- #
# Embedded background worker for in-process execution                        #
# --------------------------------------------------------------------------- #

_embedded_thread: threading.Thread | None = None
_embedded_shutdown = threading.Event()
_wake_event = threading.Event()


def wake_worker() -> None:
    """Signal the background worker loop to immediately check for queued jobs.
    Ensures the embedded worker thread is active if not in test mode.
    """
    global _embedded_thread
    import os, sys
    is_testing = (
        "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
        or os.environ.get("RUN_EMBEDDED_WORKER") == "false"
        or not getattr(settings, "RUN_EMBEDDED_WORKER", False)
    )
    if not is_testing:
        if _embedded_thread is None or not _embedded_thread.is_alive():
            try:
                start_embedded_worker()
            except Exception as start_err:
                logger.warning("Could not auto-start embedded worker: %s", start_err)

    _wake_event.set()


def _embedded_worker_loop() -> None:
    logger.info("Embedded background worker loop started (id=%s)", WORKER_ID)
    while not _embedded_shutdown.is_set():
        try:
            db = SessionLocal()
            try:
                run_once(db)
            finally:
                db.close()
        except Exception:
            logger.exception("Unhandled error in embedded worker cycle")

        # Sleep up to poll interval or wake up immediately if a new job is submitted
        timeout = float(getattr(settings, "WORKER_POLL_INTERVAL_SECONDS", 2) or 2)
        _wake_event.wait(timeout=min(timeout, 2.0))
        _wake_event.clear()

    logger.info("Embedded background worker loop terminated.")


def start_embedded_worker() -> None:
    """Start the background worker in a daemon thread if not already running."""
    global _embedded_thread
    import os, sys
    is_testing = (
        "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
        or os.environ.get("RUN_EMBEDDED_WORKER") == "false"
    )
    if is_testing:
        logger.info("Embedded worker skipped during testing/pytest.")
        return

    if _embedded_thread is not None and _embedded_thread.is_alive():
        return
    _embedded_shutdown.clear()
    _embedded_thread = threading.Thread(
        target=_embedded_worker_loop,
        name="EmbeddedWorkerThread",
        daemon=True,
    )
    _embedded_thread.start()
    logger.info("Embedded worker thread started.")


def stop_embedded_worker() -> None:
    """Stop the background worker thread gracefully."""
    global _embedded_thread
    _embedded_shutdown.set()
    _wake_event.set()
    if _embedded_thread is not None and _embedded_thread.is_alive():
        _embedded_thread.join(timeout=3.0)
    _embedded_thread = None
    logger.info("Embedded worker thread stopped.")


def main():
    logger.info("Worker starting (id=%s)", WORKER_ID)
    logger.info("Provider: %s", settings.effective_provider)
    logger.info("Poll interval: %ds", settings.WORKER_POLL_INTERVAL_SECONDS)

    while not _shutdown_requested:
        db = SessionLocal()
        try:
            count = run_once(db)
            if count == 0:
                logger.debug("No jobs this cycle, sleeping…")
        except Exception:
            logger.exception("Unhandled error in poll loop")
        finally:
            db.close()

        if not _shutdown_requested:
            time.sleep(settings.WORKER_POLL_INTERVAL_SECONDS)

    logger.info("Worker shutdown complete.")


if __name__ == "__main__":
    main()
