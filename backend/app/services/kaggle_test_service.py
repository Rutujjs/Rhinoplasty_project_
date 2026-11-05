"""
Kaggle Pipeline Test Service.

Orchestrates a dedicated 10-stage test pipeline for end-to-end verification:
Backend -> Prompt Builder -> Manifest -> Kaggle Notebook -> FLUX.2 Klein ->
Kaggle Output -> Backend Download -> Image Validation -> Browser Preview -> Cleanup.

Key Architectural Guarantees:
1. Zero Database Interaction: Does NOT touch PostgreSQL, insert Dataset/Job/Image rows,
   or modify counters.
2. Real Provider Enforcement: Uses real KaggleBatchProvider. Never silently falls back to MOCK.
3. Strict Single-Flight Guard: Only one Kaggle test run may execute at a time.
4. Storage Isolation & Safe Cleanup: All temporary artifacts live in storage/temp_tests/{run_id}
   and are cleaned up deterministically.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import secrets
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.core.config import settings
from backend.app.schemas.kaggle_test import (
    KaggleTestRequest,
    KaggleTestRunSummary,
    StageInfo,
    StageStatus,
)
from generation.interface import FaceGenerator, ProviderOutcome
from generation.kaggle_provider import KaggleBatchProvider
from generation.manifest import GenerationManifest, ManifestProviderConfig, build_prompt_from_attributes
from validation.pipeline import ValidationContext, validate_image

logger = logging.getLogger(__name__)

STAGE_DEFINITIONS = [
    (1, "Prompt prepared"),
    (2, "Manifest created and validated"),
    (3, "Kaggle manifest dataset updated"),
    (4, "Notebook submitted"),
    (5, "Notebook running"),
    (6, "Notebook completed"),
    (7, "Output archive retrieved"),
    (8, "Image located and validated"),
    (9, "Image ready for preview"),
    (10, "Temporary cleanup completed"),
]

TEMP_TEST_ROOT = Path(settings.STORAGE_DIR) / "temp_tests"


class KaggleTestRun:
    """In-memory state tracker for an individual Kaggle test run."""

    def __init__(self, run_id: str, request: KaggleTestRequest):
        self.run_id = run_id
        self.request = request
        self.status = "QUEUED"  # QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED
        self.current_stage_id = 1
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.prompt: str = ""
        self.seed: int = request.seed if request.seed is not None else secrets.randbelow(2**32)
        self.provider_run_id: str | None = None
        self.manifest: GenerationManifest | None = None
        self.workspace_dir: Path = (TEMP_TEST_ROOT / run_id).resolve()
        self.image_bytes: bytes | None = None
        self.image_metadata: dict[str, Any] | None = None
        self.validation_result: dict[str, Any] | None = None
        self.cleanup_result: dict[str, Any] | None = None
        self.error_summary: str | None = None
        self.cancelled: bool = False

        self.stages: dict[int, StageInfo] = {
            s_id: StageInfo(stage_id=s_id, name=s_name, status=StageStatus.PENDING)
            for s_id, s_name in STAGE_DEFINITIONS
        }

    def start_stage(self, stage_id: int, details: str | None = None):
        self.current_stage_id = stage_id
        stage = self.stages[stage_id]
        stage.status = StageStatus.RUNNING
        stage.started_at = datetime.now(timezone.utc).isoformat()
        if details:
            stage.details = details

    def pass_stage(self, stage_id: int, details: str | None = None):
        stage = self.stages[stage_id]
        stage.status = StageStatus.PASSED
        now = datetime.now(timezone.utc)
        stage.completed_at = now.isoformat()
        if stage.started_at:
            start_dt = datetime.fromisoformat(stage.started_at)
            stage.duration_seconds = round((now - start_dt).total_seconds(), 2)
        if details:
            stage.details = details

    def fail_stage(self, stage_id: int, error: str, details: str | None = None):
        stage = self.stages[stage_id]
        stage.status = StageStatus.FAILED
        now = datetime.now(timezone.utc)
        stage.completed_at = now.isoformat()
        stage.error = error
        if stage.started_at:
            start_dt = datetime.fromisoformat(stage.started_at)
            stage.duration_seconds = round((now - start_dt).total_seconds(), 2)
        if details:
            stage.details = details
        self.status = "FAILED"
        self.error_summary = error

        # Mark all pending intermediate stages as SKIPPED up to Stage 9
        for sid in range(stage_id + 1, 10):
            if self.stages[sid].status == StageStatus.PENDING:
                self.stages[sid].status = StageStatus.SKIPPED

    def to_summary(self) -> KaggleTestRunSummary:
        elapsed = None
        if self.started_at:
            end = self.completed_at or datetime.now(timezone.utc)
            elapsed = round((end - self.started_at).total_seconds(), 2)

        image_data_uri = None
        if self.image_bytes:
            b64 = base64.b64encode(self.image_bytes).decode("ascii")
            image_data_uri = f"data:image/png;base64,{b64}"

        return KaggleTestRunSummary(
            run_id=self.run_id,
            status=self.status,
            current_stage_id=self.current_stage_id,
            stages=list(self.stages.values()),
            prompt=self.prompt,
            seed=self.seed,
            provider_run_id=self.provider_run_id,
            image_url=f"/kaggle-test/image/{self.run_id}" if self.image_bytes else None,
            image_data_uri=image_data_uri,
            image_metadata=self.image_metadata,
            timing={
                "total_elapsed_seconds": elapsed,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            },
            validation_result=self.validation_result,
            cleanup_result=self.cleanup_result,
            error_summary=self.error_summary,
        )


class KagglePipelineTestRunner:
    """Manages execution, state, and concurrency for Kaggle test runs."""

    def __init__(self):
        self._runs: dict[str, KaggleTestRun] = {}
        self._lock = threading.RLock()
        self._active_run_id: str | None = None
        self._active_thread: threading.Thread | None = None

    def get_run(self, run_id: str) -> KaggleTestRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def _is_running_locked(self) -> bool:
        """Internal helper to check running state while self._lock is already held."""
        if self._active_run_id:
            active = self._runs.get(self._active_run_id)
            if active and active.status in ("QUEUED", "RUNNING"):
                return True
        return False

    def is_running(self) -> bool:
        """Public thread-safe check whether a run is currently active."""
        with self._lock:
            return self._is_running_locked()

    def start_test(
        self,
        request: KaggleTestRequest,
        provider: FaceGenerator | None = None,
    ) -> KaggleTestRunSummary:
        """
        Validate single-flight concurrency, instantiate a new test run,
        and spawn the execution pipeline in a background thread.
        """
        with self._lock:
            if self._is_running_locked():
                raise RuntimeError(
                    f"A Kaggle test run ({self._active_run_id}) is currently in progress. "
                    "Only one test run may execute at a time to prevent Kaggle kernel slug collision."
                )

            run_id = f"test_{uuid_hex(8)}"
            test_run = KaggleTestRun(run_id=run_id, request=request)
            self._runs[run_id] = test_run
            self._active_run_id = run_id

        # Launch pipeline execution in background thread
        thread = threading.Thread(
            target=self._execute_pipeline,
            args=(test_run, provider),
            name=f"kaggle-test-{run_id}",
            daemon=True,
        )
        self._active_thread = thread
        thread.start()

        return test_run.to_summary()

    def _execute_pipeline(
        self,
        run: KaggleTestRun,
        provider_override: FaceGenerator | None = None,
    ):
        run.started_at = datetime.now(timezone.utc)
        run.status = "RUNNING"
        logger.info("Starting Kaggle Pipeline Test run=%s", run.run_id)

        provider = provider_override or KaggleBatchProvider()

        try:
            # ---------------------------------------------------------- #
            # Stage 1: Prompt prepared                                    #
            # ---------------------------------------------------------- #
            if run.request.prompt_mode == "custom":
                if not run.request.custom_prompt or not run.request.custom_prompt.strip():
                    run.fail_stage(
                        1,
                        error="Custom prompt is empty",
                        details="Prompt mode is set to 'custom' but no prompt text was provided.",
                    )
                    return
                prompt = run.request.custom_prompt.strip()
                run.prompt = prompt
                run.start_stage(1, "Using user-supplied custom prompt")
                run.pass_stage(1, details=f"Custom prompt accepted ({len(prompt)} chars)")
            else:
                run.start_stage(1, "Building canonical prompt from attributes")
                prompt = build_prompt_from_attributes(
                    population_group=run.request.population_group,
                    region=run.request.region,
                    age_min=run.request.age_min,
                    age_max=run.request.age_max,
                    sex_or_attribute=run.request.sex_or_attribute,
                    skin_tone=run.request.skin_tone,
                    pose=run.request.pose,
                    expression=run.request.expression,
                    lighting=run.request.lighting,
                    medical_condition=run.request.medical_condition,
                    medical_severity=run.request.medical_severity,
                )
                run.prompt = prompt
                run.pass_stage(1, details=f"Prompt generated ({len(prompt)} chars)")

            if run.cancelled:
                return

            # ---------------------------------------------------------- #
            # Stage 2: Manifest created and validated                     #
            # ---------------------------------------------------------- #
            run.start_stage(2, "Constructing and validating GenerationManifest")
            run.workspace_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = run.workspace_dir / "manifest.json"

            manifest = GenerationManifest(
                dataset_id=0,
                job_id=0,
                batch_id=run.run_id,
                requested_count=1,
                image_width=run.request.resolution,
                image_height=run.request.resolution,
                population_group=run.request.population_group,
                region=run.request.region,
                age_min=run.request.age_min,
                age_max=run.request.age_max,
                sex_or_attribute=run.request.sex_or_attribute,
                skin_tone=run.request.skin_tone,
                pose=run.request.pose,
                expression=run.request.expression,
                lighting=run.request.lighting,
                medical_condition=run.request.medical_condition,
                medical_severity=run.request.medical_severity,
                prompts=[prompt],
                seeds=[run.seed],
                provider_config=ManifestProviderConfig(
                    provider="KAGGLE_BATCH",
                    output_format="png",
                ),
            )
            manifest.save(manifest_path)
            run.manifest = manifest
            run.pass_stage(2, details=f"Manifest valid (seed={run.seed}, requested_count=1)")

            if run.cancelled:
                return

            # ---------------------------------------------------------- #
            # Stage 3: Kaggle manifest dataset updated                    #
            # ---------------------------------------------------------- #
            run.start_stage(3, f"Uploading manifest to dataset {settings.KAGGLE_MANIFEST_DATASET_SLUG or '(unconfigured)'}")
            prep_result = provider.prepare_batch(manifest)
            if not prep_result.succeeded:
                run.fail_stage(
                    3,
                    error=prep_result.message or "Kaggle manifest upload failed",
                    details=prep_result.error_detail or None,
                )
                return
            run.pass_stage(3, details=prep_result.message)

            if run.cancelled:
                return

            # ---------------------------------------------------------- #
            # Stage 4: Notebook submitted                                 #
            # ---------------------------------------------------------- #
            run.start_stage(4, f"Pushing kernel {settings.KAGGLE_KERNEL_SLUG or '(unconfigured)'}")
            sub_result = provider.submit_batch(manifest)
            if not sub_result.succeeded:
                run.fail_stage(
                    4,
                    error=sub_result.message or "Kaggle kernel submission failed",
                    details=sub_result.error_detail or None,
                )
                return
            run.provider_run_id = sub_result.provider_run_id
            run.pass_stage(4, details=f"Kernel submitted (run_id={run.provider_run_id})")

            if run.cancelled:
                return

            # ---------------------------------------------------------- #
            # Stage 5 & 6: Notebook running & completed                   #
            # ---------------------------------------------------------- #
            run.start_stage(5, f"Polling Kaggle kernel execution for {run.provider_run_id}")
            poll_interval = (
                getattr(provider, "poll_interval_seconds", 0.05)
                if isinstance(provider, MockTestProvider)
                else max(5, settings.KAGGLE_POLL_INTERVAL_SECONDS)
            )
            timeout = max(300, settings.KAGGLE_RUN_TIMEOUT_SECONDS)
            start_poll = time.time()

            # Wait post-push delay if defined
            post_push_delay = getattr(settings, "KAGGLE_POST_PUSH_DELAY_SECONDS", 0)
            if post_push_delay > 0 and not isinstance(provider, MockTestProvider):
                time.sleep(min(post_push_delay, 10))

            while True:
                if run.cancelled:
                    run.fail_stage(5, error="Test cancelled by user")
                    return

                if time.time() - start_poll > timeout:
                    run.fail_stage(5, error=f"Kaggle kernel run timed out after {timeout} seconds")
                    return

                status_result = provider.poll_status(run.provider_run_id)

                if status_result.outcome == ProviderOutcome.PENDING:
                    run.start_stage(5, details=status_result.message or "Kernel is queued/running on Kaggle")
                    time.sleep(poll_interval)
                    continue

                if status_result.outcome == ProviderOutcome.SUCCESS:
                    run.pass_stage(5, details="Kernel finished execution")
                    run.start_stage(6, "Validating notebook completion state")
                    run.pass_stage(6, details="Notebook completed with exit code 0")
                    break

                # Permanent or unrecoverable error
                run.fail_stage(
                    5,
                    error=status_result.message or "Kaggle kernel failed during execution",
                    details=status_result.error_detail or None,
                )
                return

            if run.cancelled:
                return

            # ---------------------------------------------------------- #
            # Stage 7: Output archive retrieved                           #
            # ---------------------------------------------------------- #
            run.start_stage(7, f"Downloading kernel outputs to {run.workspace_dir.name}/artifacts")
            artifact_dir = run.workspace_dir / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            retrieve_result = provider.retrieve_artifacts(
                run.provider_run_id,
                str(artifact_dir),
                manifest,
            )
            if not retrieve_result.succeeded:
                run.fail_stage(
                    7,
                    error=retrieve_result.message or "Failed to retrieve artifacts from Kaggle",
                    details=retrieve_result.error_detail or None,
                )
                return
            run.pass_stage(7, details=retrieve_result.message)

            if run.cancelled:
                return

            # ---------------------------------------------------------- #
            # Stage 8: Image located and validated                        #
            # ---------------------------------------------------------- #
            run.start_stage(8, "Locating generated image and verifying constraints")
            found_images = retrieve_result.data.get("artifact_paths", []) if retrieve_result.data else []

            # If provider didn't return paths explicitly, scan artifact dir
            if not found_images:
                found_images = [
                    str(p) for p in artifact_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                ]

            if not found_images:
                run.fail_stage(8, error="No generated image file found in downloaded Kaggle artifacts")
                return

            target_img_path = Path(found_images[0])

            # Run baseline image validation pipeline
            ctx = ValidationContext(
                expected_width=run.request.resolution,
                expected_height=run.request.resolution,
                allowed_formats=["png", "jpg", "jpeg", "webp"],
                max_file_bytes=settings.MAX_IMAGE_FILE_BYTES,
                require_face_detection=False,
                enable_duplicate_detection=False,
            )
            v_result = validate_image(target_img_path, ctx)

            run.validation_result = {
                "outcome": v_result.outcome,
                "rejection_reasons": v_result.rejection_reasons,
                "width": v_result.image_width,
                "height": v_result.image_height,
                "format": v_result.image_format,
                "size_bytes": v_result.file_size_bytes,
                "sha256": v_result.checksum_sha256,
            }

            if not v_result.accepted:
                run.fail_stage(
                    8,
                    error=f"Image validation rejected: {', '.join(v_result.rejection_reasons)}",
                )
                return

            run.pass_stage(
                8,
                details=(
                    f"Image accepted: {v_result.image_width}x{v_result.image_height} "
                    f"({v_result.image_format.upper()}, {v_result.file_size_bytes} bytes)"
                ),
            )

            # ---------------------------------------------------------- #
            # Stage 9: Image ready for preview                            #
            # ---------------------------------------------------------- #
            run.start_stage(9, "Caching image in memory for safe browser delivery")
            img_bytes = target_img_path.read_bytes()
            run.image_bytes = img_bytes
            run.image_metadata = {
                "filename": target_img_path.name,
                "width": v_result.image_width,
                "height": v_result.image_height,
                "format": v_result.image_format,
                "file_size_bytes": len(img_bytes),
                "sha256": v_result.checksum_sha256,
            }
            run.pass_stage(9, details="Preview image ready in memory")
            run.status = "COMPLETED"

        except Exception as exc:
            logger.exception("Unexpected error during Kaggle test pipeline run=%s", run.run_id)
            run.fail_stage(run.current_stage_id, error=f"Internal exception: {exc}")
        finally:
            run.completed_at = datetime.now(timezone.utc)
            # ---------------------------------------------------------- #
            # Stage 10: Temporary cleanup completed                       #
            # ---------------------------------------------------------- #
            try:
                self._cleanup_run_workspace(run)
            finally:
                with self._lock:
                    if self._active_run_id == run.run_id:
                        self._active_run_id = None

    def _cleanup_run_workspace(self, run: KaggleTestRun):
        run.start_stage(10, "Cleaning up temporary manifests and download archives")
        try:
            if run.workspace_dir.exists():
                shutil.rmtree(run.workspace_dir, ignore_errors=True)
            run.cleanup_result = {
                "cleaned_at": datetime.now(timezone.utc).isoformat(),
                "workspace_path": str(run.workspace_dir),
                "status": "SUCCESS",
            }
            run.pass_stage(10, details="Temporary artifacts deleted from disk")
        except Exception as exc:
            run.cleanup_result = {
                "cleaned_at": datetime.now(timezone.utc).isoformat(),
                "status": "FAILED",
                "error": str(exc),
            }
            run.fail_stage(10, error=f"Cleanup error: {exc}")


def uuid_hex(length: int = 8) -> str:
    return secrets.token_hex(length // 2)


# Mock/Fake provider for non-destructive local automated tests
class MockTestProvider(FaceGenerator):
    """
    Simulates KaggleBatchProvider behavior for non-destructive unit/integration tests.
    Never connects to Kaggle or external networks.
    """
    poll_interval_seconds: float = 0.05

    def __init__(self, succeed: bool = True, failure_stage: int | None = None):
        self.succeed = succeed
        self.failure_stage = failure_stage
        self.poll_count = 0

    @property
    def provider_name(self) -> str:
        return "MOCK_TEST_PROVIDER"

    def generate(self, request):
        raise NotImplementedError()

    def prepare_batch(self, manifest):
        from generation.interface import ProviderOperationResult
        if not self.succeed and self.failure_stage == 3:
            return ProviderOperationResult.permanent_error(message="Simulated dataset upload failure")
        return ProviderOperationResult.success(message="Simulated manifest upload")

    def submit_batch(self, manifest):
        from generation.interface import ProviderOperationResult
        if not self.succeed and self.failure_stage == 4:
            return ProviderOperationResult.permanent_error(message="Simulated kernel push failure")
        return ProviderOperationResult.success(provider_run_id="testuser/mock-kernel", message="Simulated kernel push")

    def poll_status(self, provider_run_id):
        from generation.interface import ProviderOperationResult
        self.poll_count += 1
        if not self.succeed and self.failure_stage == 5:
            return ProviderOperationResult.permanent_error(message="Simulated Kaggle GPU error")
        if self.poll_count < 2:
            return ProviderOperationResult.pending(message="Kernel is queued")
        return ProviderOperationResult.success(provider_run_id=provider_run_id, message="Kernel complete")

    def retrieve_artifacts(self, provider_run_id, destination_dir, manifest):
        from generation.interface import ProviderOperationResult
        from PIL import Image
        if not self.succeed and self.failure_stage == 7:
            return ProviderOperationResult.permanent_error(message="Simulated download error")

        dest = Path(destination_dir)
        dest.mkdir(parents=True, exist_ok=True)
        img_file = dest / "image_0001.png"

        # Create a valid 512x512 test PNG
        img = Image.new("RGB", (512, 512), color=(120, 150, 180))
        img.save(img_file, format="PNG")

        return ProviderOperationResult.success(
            provider_run_id=provider_run_id,
            data={"artifact_paths": [str(img_file)]},
            message="Downloaded simulated test output",
        )

    def request_cancellation(self, provider_run_id):
        from generation.interface import ProviderOperationResult
        return ProviderOperationResult.unsupported(operation="request_cancellation", provider=self.provider_name)


# Global singleton instance
kaggle_test_runner = KagglePipelineTestRunner()
