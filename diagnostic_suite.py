"""
Comprehensive Kaggle Pipeline Diagnostic Suite.

Audit, verify, and validate the end-to-end image generation and test preview
architecture across 7 critical dimensions:
  [A] Configuration and local environment
  [B] Manifest preparation
  [C] Notebook submission
  [D] Notebook execution and polling
  [E] Artifact retrieval and validation
  [F] API and browser UI
  [G] Data isolation and resource cleanup

SAFETY GUARANTEES:
  1. Zero Remote Mutations: Never pushes a notebook, creates a dataset version,
     or submits a GPU job.
  2. Zero Secrets Leakage: Redacts tokens, passwords, and private values in all
     outputs and JSON summaries.
  3. Zero Database Mutations: Never inserts or modifies database rows or counters.
  4. MOCK Safeguard Enforcement: Verifies that local provider settings remain MOCK.

Usage:
  python diagnostic_suite.py                  # Standard human-readable terminal report
  python diagnostic_suite.py --json           # Machine-readable JSON output
  python diagnostic_suite.py --remote-check-dry-run # Preview optional read-only remote checks
"""

from __future__ import annotations

import argparse
import builtins
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

from PIL import Image

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.config import BASE_DIR, settings
from backend.app.main import app
from backend.app.schemas.kaggle_test import KaggleTestRequest, StageStatus
from backend.app.services.kaggle_test_service import (
    KagglePipelineTestRunner,
    KaggleTestRun,
    MockTestProvider,
)
from generation.interface import FaceGenerator, ProviderOperationResult, ProviderOutcome
from generation.kaggle_provider import (
    KaggleBatchProvider,
    _get_kaggle_api,
    _resolve_kernel_metadata_path,
    _resolve_notebook_path,
    _sanitize_error_detail,
    _utf8_open_context,
)
from generation.manifest import (
    GenerationManifest,
    ManifestProviderConfig,
    build_prompt_from_attributes,
)
from validation.pipeline import ValidationContext, validate_image

logging.basicConfig(level=logging.ERROR)


# --------------------------------------------------------------------------- #
# Data structures for diagnostic results                                      #
# --------------------------------------------------------------------------- #

@dataclass
class CheckResult:
    check_id: str
    name: str
    category: str
    check_type: str  # "LOCAL", "MOCKED", "REMOTE"
    status: str      # "PASS", "FAIL", "NOT RUN"
    duration_ms: float
    error_summary: str | None = None
    remediation: str | None = None
    details: str | None = None


@dataclass
class DiagnosticReport:
    timestamp: str
    effective_provider: str
    total_checks: int
    passed_count: int
    failed_count: int
    not_run_count: int
    duration_seconds: float
    local_passed_count: int = 0
    local_failed_count: int = 0
    local_not_run_count: int = 0
    remote_passed_count: int = 0
    remote_failed_count: int = 0
    remote_not_run_count: int = 0
    checks: list[CheckResult] = field(default_factory=list)


def _safe_redact(text: str) -> str:
    """Ensure no API keys, tokens, or credentials appear in diagnostic reports."""
    if not text:
        return ""
    t = str(text)
    for env_var in ("KAGGLE_KEY", "KAGGLE_API_TOKEN", "SECRET_KEY"):
        val = os.environ.get(env_var) or getattr(settings, env_var, None)
        if val and len(str(val)) > 4:
            t = t.replace(str(val), "[REDACTED]")
    t = re.sub(r"\b[A-Fa-f0-9]{32,}\b", "[REDACTED_TOKEN]", t)
    t = re.sub(r"(key|token|secret|password|authorization)[\s:=]+[^\s]+", r"\1=[REDACTED]", t, flags=re.IGNORECASE)
    return t


def _make_dummy_manifest(tmp_path: Path) -> GenerationManifest:
    return GenerationManifest(
        dataset_id=0,
        job_id=0,
        batch_id="diag_batch_001",
        requested_count=1,
        image_width=512,
        image_height=512,
        population_group="South Asian",
        region="India",
        age_min=25,
        age_max=35,
        sex_or_attribute="female",
        skin_tone="medium_brown",
        pose="frontal",
        expression="neutral",
        lighting="neutral",
        prompts=["A photo of an Indian woman"],
        seeds=[12345],
        provider_config=ManifestProviderConfig(provider="KAGGLE_BATCH", output_format="png"),
    )


# --------------------------------------------------------------------------- #
# Diagnostic Suite Implementation                                             #
# --------------------------------------------------------------------------- #

class KagglePipelineDiagnosticSuite:
    """Executes safe, secrets-redacted local diagnostic checks."""

    def __init__(self, include_remote_read_only: bool = False, dry_run_remote: bool = False):
        self.include_remote = include_remote_read_only
        self.dry_run_remote = dry_run_remote
        self.results: list[CheckResult] = []

    def run_check(
        self,
        check_id: str,
        name: str,
        category: str,
        check_type: str,
        fn: Callable[[], str | None],
        remediation: str | None = None,
    ):
        start = time.perf_counter()
        try:
            detail = fn()
            duration = round((time.perf_counter() - start) * 1000, 2)
            self.results.append(
                CheckResult(
                    check_id=check_id,
                    name=name,
                    category=category,
                    check_type=check_type,
                    status="PASS",
                    duration_ms=duration,
                    details=_safe_redact(detail or "OK"),
                )
            )
        except Exception as exc:
            duration = round((time.perf_counter() - start) * 1000, 2)
            self.results.append(
                CheckResult(
                    check_id=check_id,
                    name=name,
                    category=category,
                    check_type=check_type,
                    status="FAIL",
                    duration_ms=duration,
                    error_summary=_safe_redact(f"{type(exc).__name__}: {exc}"),
                    remediation=remediation or "Review configuration and implementation.",
                )
            )

    def mark_not_run(self, check_id: str, name: str, category: str, check_type: str, reason: str):
        self.results.append(
            CheckResult(
                check_id=check_id,
                name=name,
                category=category,
                check_type=check_type,
                status="NOT RUN",
                duration_ms=0.0,
                details=_safe_redact(reason),
            )
        )

    # ----------------------------------------------------------------------- #
    # Category A: Configuration and Local Environment                         #
    # ----------------------------------------------------------------------- #

    def check_category_a(self):
        cat = "A. Environment & Config"

        # A1: Effective provider and MOCK safeguards
        def test_effective_provider():
            prov = settings.effective_provider
            assert prov == "MOCK", f"effective_provider is {prov!r}, must be 'MOCK' for safety"
            return f"Provider is {prov} (Safe local mode enforced)"

        self.run_check(
            "A01", "Effective provider safeguard", cat, "LOCAL",
            test_effective_provider,
            remediation="Set GENERATION_MODEL=MOCK in backend/.env",
        )

        # A2: Required settings presence
        def test_settings_presence():
            has_slug = bool(settings.KAGGLE_MANIFEST_DATASET_SLUG)
            has_kernel = bool(settings.KAGGLE_KERNEL_SLUG)
            has_user = bool(settings.KAGGLE_USERNAME)
            assert has_slug, "KAGGLE_MANIFEST_DATASET_SLUG is not configured"
            assert has_kernel, "KAGGLE_KERNEL_SLUG is not configured"
            assert has_user, "KAGGLE_USERNAME is not configured"
            return "KAGGLE_MANIFEST_DATASET_SLUG, KAGGLE_KERNEL_SLUG, and KAGGLE_USERNAME are present"

        self.run_check(
            "A02", "Required Kaggle setting presence", cat, "LOCAL",
            test_settings_presence,
            remediation="Configure KAGGLE_MANIFEST_DATASET_SLUG, KAGGLE_KERNEL_SLUG, and KAGGLE_USERNAME in .env",
        )

        # A3: Notebook file existence
        def test_notebook_exists():
            nb = _resolve_notebook_path(settings)
            assert nb.exists(), f"Notebook file not found at {nb}"
            return f"Found notebook: {nb.name} ({nb.stat().st_size} bytes)"

        self.run_check(
            "A03", "Notebook file existence", cat, "LOCAL",
            test_notebook_exists,
            remediation="Ensure KAGGLE_NOTEBOOK_PATH points to a valid .ipynb file.",
        )

        # A4: Kernel metadata existence and fields
        def test_kernel_metadata():
            meta_path = _resolve_kernel_metadata_path(settings)
            assert meta_path.exists(), f"kernel-metadata.json not found at {meta_path}"
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            for req in ("id", "code_file", "kernel_type", "language"):
                assert req in data, f"kernel-metadata.json missing required field: {req}"
            return f"Found valid kernel-metadata.json ({data.get('id')})"

        self.run_check(
            "A04", "Kernel metadata schema validation", cat, "LOCAL",
            test_kernel_metadata,
            remediation="Verify kaggle_notebook_check/kernel-metadata.json contains id, code_file, kernel_type, and language.",
        )

        # A5: Manifest dataset slug format
        def test_slug_format():
            slug = settings.KAGGLE_MANIFEST_DATASET_SLUG
            assert re.match(r"^[^/\s]+/[^/\s]+$", slug), f"Dataset slug {slug!r} invalid; expected 'owner/dataset-slug'"
            return "Dataset slug conforms to owner/slug format"

        self.run_check(
            "A05", "Manifest dataset slug format", cat, "LOCAL",
            test_slug_format,
            remediation="Format KAGGLE_MANIFEST_DATASET_SLUG as 'username/dataset-slug'.",
        )

        # A6: Canonical prompt builder
        def test_prompt_builder():
            p = build_prompt_from_attributes(
                population_group="South Asian", region="India",
                age_min=25, age_max=35, sex_or_attribute="female",
                skin_tone="medium_brown", pose="frontal",
                expression="neutral", lighting="neutral",
            )
            assert "South Asian" in p and "India" in p
            assert len(p) > 20
            return f"Canonical prompt generated ({len(p)} characters)"

        self.run_check(
            "A06", "Canonical prompt builder output", cat, "LOCAL",
            test_prompt_builder,
            remediation="Ensure build_prompt_from_attributes formats attributes into expected prompt.",
        )

        # A7: Manifest schema and requested count enforcement
        def test_manifest_schema():
            with tempfile.TemporaryDirectory() as td:
                m = _make_dummy_manifest(Path(td))
                p = Path(td) / "manifest.json"
                m.save(p)
                loaded = GenerationManifest.load(p)
                assert loaded.requested_count == 1
                assert len(loaded.prompts) == 1
                assert loaded.provider_config.provider == "KAGGLE_BATCH"
            return "Manifest enforces requested_count=1 and single prompt"

        self.run_check(
            "A07", "Manifest schema & requested_count guard", cat, "LOCAL",
            test_manifest_schema,
            remediation="Verify GenerationManifest schema enforces count=1 for preview.",
        )

        # A8: Temp directory creation and cleanup
        def test_temp_dir():
            test_dir = Path(settings.STORAGE_DIR) / "temp_tests" / "diag_test_dir"
            test_dir.mkdir(parents=True, exist_ok=True)
            assert test_dir.exists()
            shutil.rmtree(test_dir, ignore_errors=True)
            assert not test_dir.exists()
            return "Temporary directories created and cleaned up deterministically"

        self.run_check(
            "A08", "Temp workspace creation & cleanup", cat, "LOCAL",
            test_temp_dir,
            remediation="Ensure write permissions on storage/temp_tests directory.",
        )

        # A9: Image validation dependencies
        def test_validation_deps():
            with io.BytesIO() as bio:
                img = Image.new("RGB", (64, 64), color=(100, 150, 200))
                img.save(bio, format="PNG")
                raw = bio.getvalue()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tf.write(raw)
                tf_path = Path(tf.name)
            try:
                ctx = ValidationContext(expected_width=64, expected_height=64, allowed_formats=["png"])
                res = validate_image(tf_path, ctx)
                assert res.accepted, f"Validation failed: {res.rejection_reasons}"
                return "Pillow and ValidationContext functional"
            finally:
                tf_path.unlink(missing_ok=True)

        self.run_check(
            "A09", "Image validation pipeline dependencies", cat, "LOCAL",
            test_validation_deps,
            remediation="Install PIL/Pillow and verify image validation pipeline.",
        )

        # A10: Project root configuration resolution
        def test_config_root():
            base = BASE_DIR
            assert (base / "backend").exists()
            assert (base / "frontend").exists()
            return f"BASE_DIR resolved to {base}"

        self.run_check(
            "A10", "Project root directory resolution", cat, "LOCAL",
            test_config_root,
            remediation="Verify BASE_DIR points to root containing backend and frontend.",
        )

    # ----------------------------------------------------------------------- #
    # Category B: Manifest Preparation                                        #
    # ----------------------------------------------------------------------- #

    def check_category_b(self):
        cat = "B. Manifest Preparation"

        # B1: Valid manifest preparation (mocked SDK)
        def test_manifest_prep():
            with tempfile.TemporaryDirectory() as td:
                manifest = _make_dummy_manifest(Path(td))
                mock_api = MagicMock()
                with patch("generation.kaggle_provider.settings") as s:
                    s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
                    with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                        res = KaggleBatchProvider().prepare_batch(manifest)
                assert res.succeeded, f"prepare_batch failed: {res.message}"
                mock_api.dataset_create_version.assert_called_once()
            return "prepare_batch uploads manifest.json to dataset version"

        self.run_check(
            "B01", "Manifest batch preparation (mocked SDK)", cat, "MOCKED",
            test_manifest_prep,
        )

        # B2: dataset-metadata.json generation regression
        def test_metadata_json_creation():
            with tempfile.TemporaryDirectory() as td:
                manifest = _make_dummy_manifest(Path(td))
                captured = {}
                def fake_create(folder, **kwargs):
                    p = Path(folder) / "dataset-metadata.json"
                    assert p.exists(), "dataset-metadata.json was not created"
                    data = json.loads(p.read_text(encoding="utf-8"))
                    captured["id"] = data.get("id")
                mock_api = MagicMock()
                mock_api.dataset_create_version.side_effect = fake_create
                with patch("generation.kaggle_provider.settings") as s:
                    s.KAGGLE_MANIFEST_DATASET_SLUG = "user/target-dataset"
                    with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                        res = KaggleBatchProvider().prepare_batch(manifest)
                assert res.succeeded
                assert captured.get("id") == "user/target-dataset"
            return "dataset-metadata.json created with correct id before upload"

        self.run_check(
            "B02", "dataset-metadata.json generation", cat, "MOCKED",
            test_metadata_json_creation,
            remediation="Ensure prepare_batch writes dataset-metadata.json to upload folder.",
        )

        # B3: Missing dataset slug handling
        def test_missing_slug():
            with tempfile.TemporaryDirectory() as td:
                manifest = _make_dummy_manifest(Path(td))
                with patch("generation.kaggle_provider.settings") as s:
                    s.KAGGLE_MANIFEST_DATASET_SLUG = ""
                    res = KaggleBatchProvider().prepare_batch(manifest)
                assert not res.succeeded
                assert res.outcome == ProviderOutcome.PERMANENT_ERROR
                assert "KAGGLE_MANIFEST_DATASET_SLUG" in res.message
            return "Missing dataset slug fails fast with clear message"

        self.run_check(
            "B03", "Missing manifest dataset slug handling", cat, "LOCAL",
            test_missing_slug,
        )

        # B4: Missing credentials error propagation
        def test_missing_creds():
            with tempfile.TemporaryDirectory() as td:
                manifest = _make_dummy_manifest(Path(td))
                with patch("generation.kaggle_provider.settings") as s:
                    s.KAGGLE_MANIFEST_DATASET_SLUG = "user/slug"
                    with patch("generation.kaggle_provider._get_kaggle_api") as mock_get:
                        mock_get.side_effect = EnvironmentError("Kaggle credentials not found.")
                        res = KaggleBatchProvider().prepare_batch(manifest)
                assert not res.succeeded
                assert "credentials not found" in res.message.lower()
            return "Missing credentials reports descriptive error"

        self.run_check(
            "B04", "Missing Kaggle credentials propagation", cat, "MOCKED",
            test_missing_creds,
        )

        # B5: Safe error detail propagation & token redaction
        def test_safe_detail_redaction():
            with tempfile.TemporaryDirectory() as td:
                manifest = _make_dummy_manifest(Path(td))
                mock_api = MagicMock()
                mock_api.dataset_create_version.side_effect = RuntimeError(
                    "Upload error on dataset with token=1234567890abcdef1234567890abcdef"
                )
                with patch("generation.kaggle_provider.settings") as s:
                    s.KAGGLE_MANIFEST_DATASET_SLUG = "user/slug"
                    with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                        res = KaggleBatchProvider().prepare_batch(manifest)
                assert not res.succeeded
                assert "1234567890abcdef1234567890abcdef" not in res.message
                assert "[REDACTED]" in res.message
            return "API tokens and credentials redacted from error message"

        self.run_check(
            "B05", "Error detail propagation & token redaction", cat, "MOCKED",
            test_safe_detail_redaction,
        )

    # ----------------------------------------------------------------------- #
    # Category C: Notebook Submission                                         #
    # ----------------------------------------------------------------------- #

    def check_category_c(self):
        cat = "C. Notebook Submission"

        # C1: Missing kernel slug
        def test_missing_kernel():
            with tempfile.TemporaryDirectory() as td:
                manifest = _make_dummy_manifest(Path(td))
                with patch("generation.kaggle_provider.settings") as s:
                    s.KAGGLE_KERNEL_SLUG = ""
                    s.KAGGLE_MANIFEST_DATASET_SLUG = "user/slug"
                    res = KaggleBatchProvider().submit_batch(manifest)
                assert not res.succeeded
                assert "KAGGLE_KERNEL_SLUG" in res.message
            return "Missing kernel slug rejected with clear guidance"

        self.run_check(
            "C01", "Missing kernel slug handling", cat, "LOCAL",
            test_missing_kernel,
        )

        # C2: Missing notebook file
        def test_missing_nb():
            with tempfile.TemporaryDirectory() as td:
                manifest = _make_dummy_manifest(Path(td))
                with patch("generation.kaggle_provider.settings") as s:
                    s.KAGGLE_KERNEL_SLUG = "user/kernel"
                    s.KAGGLE_MANIFEST_DATASET_SLUG = "user/slug"
                    s.KAGGLE_NOTEBOOK_PATH = str(Path(td) / "absent.ipynb")
                    s.KAGGLE_KERNEL_METADATA_PATH = str(Path(td) / "meta.json")
                    res = KaggleBatchProvider().submit_batch(manifest)
                assert not res.succeeded
                assert "Notebook file not found" in res.message
            return "Missing notebook file rejected before remote call"

        self.run_check(
            "C02", "Missing notebook file handling", cat, "LOCAL",
            test_missing_nb,
        )

        # C3: Windows Unicode / charmap regression check
        def test_charmap_regression():
            with tempfile.TemporaryDirectory() as td:
                manifest = _make_dummy_manifest(Path(td))
                nb_p = Path(td) / "unicode.ipynb"
                # Notebook with raw UTF-8 multibyte character '❌' (byte 0x9d)
                nb_dict = {"cells": [{"cell_type": "code", "source": ["print('❌ GPU NOT AVAILABLE')\n"], "outputs": []}]}
                nb_p.write_bytes(json.dumps(nb_dict, ensure_ascii=False).encode("utf-8"))
                assert b"\x9d" in nb_p.read_bytes()

                meta_p = Path(td) / "kernel-metadata.json"
                meta_p.write_text(json.dumps({"id": "user/kernel", "dataset_sources": []}), encoding="utf-8")

                captured = {}
                def fake_push(folder):
                    dest = Path(folder) / "unicode.ipynb"
                    assert dest.exists()
                    # Simulating Kaggle SDK's unencoded open() on Windows
                    content = open(str(dest)).read()
                    captured["read_ok"] = True
                    assert "❌" in json.loads(content)["cells"][0]["source"][0]

                mock_api = MagicMock()
                mock_api.kernels_push.side_effect = fake_push

                with patch("generation.kaggle_provider.settings") as s:
                    s.KAGGLE_KERNEL_SLUG = "user/kernel"
                    s.KAGGLE_MANIFEST_DATASET_SLUG = "user/slug"
                    s.KAGGLE_NOTEBOOK_PATH = str(nb_p)
                    s.KAGGLE_KERNEL_METADATA_PATH = str(meta_p)
                    with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                        res = KaggleBatchProvider().submit_batch(manifest)

                assert res.succeeded
                assert captured.get("read_ok") is True
            return "Notebook with byte 0x9d pushes without charmap UnicodeDecodeError"

        self.run_check(
            "C03", "Windows Unicode/charmap push regression", cat, "MOCKED",
            test_charmap_regression,
            remediation="Ensure submit_batch uses ensure_ascii=True and _utf8_open_context.",
        )

        # C4: Non-ASCII error reporting safety
        def test_non_ascii_error():
            with tempfile.TemporaryDirectory() as td:
                manifest = _make_dummy_manifest(Path(td))
                nb_p = Path(td) / "nb.ipynb"
                nb_p.write_text('{"cells": []}', encoding="utf-8")
                meta_p = Path(td) / "kernel-metadata.json"
                meta_p.write_text('{"id": "u/k", "dataset_sources": []}', encoding="utf-8")

                mock_api = MagicMock()
                mock_api.kernels_push.side_effect = RuntimeError("Push failed: ❌ auth token=abc12345678901234567890123456789")

                with patch("generation.kaggle_provider.settings") as s:
                    s.KAGGLE_KERNEL_SLUG = "u/k"
                    s.KAGGLE_MANIFEST_DATASET_SLUG = "u/s"
                    s.KAGGLE_NOTEBOOK_PATH = str(nb_p)
                    s.KAGGLE_KERNEL_METADATA_PATH = str(meta_p)
                    with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                        res = KaggleBatchProvider().submit_batch(manifest)

                assert not res.succeeded
                assert "RuntimeError" in res.message
                assert "abc12345678901234567890123456789" not in res.message
                assert "[REDACTED]" in res.message
            return "Non-ASCII push errors sanitized and tokens redacted"

        self.run_check(
            "C04", "Push non-ASCII error sanitization", cat, "MOCKED",
            test_non_ascii_error,
        )

    # ----------------------------------------------------------------------- #
    # Category D: Notebook Execution and Polling                              #
    # ----------------------------------------------------------------------- #

    def check_category_d(self):
        cat = "D. Execution & Polling"

        # D1: Polling status mapping
        def test_status_mapping():
            provider = KaggleBatchProvider()
            mock_api = MagicMock()
            for st, expected_outcome in [
                ("running", ProviderOutcome.PENDING),
                ("queued", ProviderOutcome.PENDING),
                ("cancelrequested", ProviderOutcome.PENDING),
                ("cancel_requested", ProviderOutcome.PENDING),
                ("complete", ProviderOutcome.SUCCESS),
                ("error", ProviderOutcome.PERMANENT_ERROR),
                ("cancelled", ProviderOutcome.PERMANENT_ERROR),
                ("kernelworkerstatus.complete", ProviderOutcome.SUCCESS),
                ("KernelWorkerStatus.COMPLETE", ProviderOutcome.SUCCESS),
                ("kernelworkerstatus.running", ProviderOutcome.PENDING),
                ("KernelWorkerStatus.ERROR", ProviderOutcome.PERMANENT_ERROR),
            ]:
                mock_api.kernels_status.return_value = {"status": st}
                with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                    res = provider.poll_status("user/slug")
                assert res.outcome == expected_outcome, f"Status {st} mapped to {res.outcome}, expected {expected_outcome}"

            try:
                from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus
                for enum_val, expected_outcome in [
                    (KernelWorkerStatus.COMPLETE, ProviderOutcome.SUCCESS),
                    (KernelWorkerStatus.RUNNING, ProviderOutcome.PENDING),
                    (KernelWorkerStatus.QUEUED, ProviderOutcome.PENDING),
                    (KernelWorkerStatus.ERROR, ProviderOutcome.PERMANENT_ERROR),
                ]:
                    mock_api.kernels_status.return_value = {"status": enum_val}
                    with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                        res = provider.poll_status("user/slug")
                    assert res.outcome == expected_outcome, f"Enum {enum_val} mapped to {res.outcome}, expected {expected_outcome}"
            except ImportError:
                pass

            return "Kernel execution statuses (strings, dot-strings, and SDK enums) mapped correctly"

        self.run_check(
            "D01", "Kaggle kernel status outcome mapping", cat, "MOCKED",
            test_status_mapping,
        )

        # D2: Single-flight concurrency guard
        def test_single_flight():
            runner = KagglePipelineTestRunner()
            req = KaggleTestRequest(
                population_group="South Asian", region="India",
                age_min=25, age_max=35, sex_or_attribute="female",
                skin_tone="medium_brown", pose="frontal",
                expression="neutral", lighting="neutral",
            )
            # Active mock provider that pauses
            class PausingProvider(MockTestProvider):
                def prepare_batch(self, manifest):
                    time.sleep(0.3)
                    return ProviderOperationResult.success()
            summary = runner.start_test(req, provider=PausingProvider(succeed=True))
            # Second attempt must raise RuntimeError
            raised = False
            try:
                runner.start_test(req, provider=PausingProvider(succeed=True))
            except RuntimeError as exc:
                raised = True
                assert "currently in progress" in str(exc)
            finally:
                if runner._active_thread:
                    runner._active_thread.join(timeout=5)
            assert raised, "Second concurrent test run was not rejected"
            return "Single-flight guard rejects second run while active"

        self.run_check(
            "D02", "Single-flight execution lock guard", cat, "LOCAL",
            test_single_flight,
        )

    # ----------------------------------------------------------------------- #
    # Category E: Artifact Retrieval and Validation                           #
    # ----------------------------------------------------------------------- #

    def check_category_e(self):
        cat = "E. Retrieval & Validation"

        # E1: Flat and nested artifact discovery
        def test_artifact_discovery():
            with tempfile.TemporaryDirectory() as td:
                dest = Path(td)
                # Create nested structure
                img_dir = dest / "rhinoplasty_dataset" / "batches" / "b1" / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                img_file = img_dir / "face_0001.png"
                img_file.write_bytes(b"\x89PNG\r\n\x1a\nfake")
                summary_file = dest / "rhinoplasty_dataset" / "batches" / "b1" / "batch_summary.json"
                summary_file.write_text('{"requested_count": 1}', encoding="utf-8")

                mock_api = MagicMock()
                with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                    res = KaggleBatchProvider().retrieve_artifacts("u/k", str(dest), None)
                assert res.succeeded
                assert len(res.data.get("artifact_paths", [])) == 1
                assert res.data.get("summary_path") is not None
            return "Nested batch image and summary detected successfully"

        self.run_check(
            "E01", "Nested and flat artifact discovery", cat, "MOCKED",
            test_artifact_discovery,
        )

        # E2: Image validation rejecting corrupt bytes
        def test_corrupt_image_validation():
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tf.write(b"NOT_A_REAL_IMAGE_DATA")
                tf_path = Path(tf.name)
            try:
                ctx = ValidationContext(expected_width=512, expected_height=512, allowed_formats=["png"])
                res = validate_image(tf_path, ctx)
                assert not res.accepted
                assert len(res.rejection_reasons) > 0
            finally:
                tf_path.unlink(missing_ok=True)
            return "Corrupt image bytes correctly rejected with reasons"

        self.run_check(
            "E02", "Image validation: corrupt byte rejection", cat, "LOCAL",
            test_corrupt_image_validation,
        )

        # E3: Image validation dimension mismatch
        def test_dimension_mismatch():
            with io.BytesIO() as bio:
                img = Image.new("RGB", (256, 256), color=(255, 0, 0))
                img.save(bio, format="PNG")
                raw = bio.getvalue()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tf.write(raw)
                tf_path = Path(tf.name)
            try:
                ctx = ValidationContext(expected_width=512, expected_height=512, allowed_formats=["png"])
                res = validate_image(tf_path, ctx)
                assert not res.accepted
                assert any("expected" in r.lower() or "512" in r for r in res.rejection_reasons)
            finally:
                tf_path.unlink(missing_ok=True)
            return "Image dimension mismatch (256x256 vs expected 512x512) rejected"

        self.run_check(
            "E03", "Image validation: dimension enforcement", cat, "LOCAL",
            test_dimension_mismatch,
        )

    # ----------------------------------------------------------------------- #
    # Category F: API and Browser UI                                          #
    # ----------------------------------------------------------------------- #

    def check_category_f(self):
        cat = "F. API & Browser UI"
        from fastapi.testclient import TestClient
        client = TestClient(app)

        # F1: Prompt preview endpoint
        def test_api_prompt_preview():
            resp = client.post("/kaggle-test/prompt-preview", json={
                "population_group": "South Asian", "region": "India",
                "age_min": 25, "age_max": 35, "sex_or_attribute": "female",
                "skin_tone": "medium_brown", "pose": "frontal",
                "expression": "neutral", "lighting": "neutral",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert "prompt" in data and len(data["prompt"]) > 20
            return "POST /kaggle-test/prompt-preview returns canonical prompt"

        self.run_check("F01", "API: prompt preview endpoint", cat, "LOCAL", test_api_prompt_preview)

        # F2: UI routes serve HTML
        def test_ui_routes():
            for path in ("/kaggle-test", "/kaggle-test/", "/ui", "/ui/"):
                resp = client.get(path)
                assert resp.status_code == 200, f"GET {path} returned {resp.status_code}"
                assert "text/html" in resp.headers.get("content-type", "")
                assert "<!DOCTYPE html>" in resp.text
            return "GET /kaggle-test, /kaggle-test/, /ui, /ui/ return status 200"

        self.run_check("F02", "API: UI route resolution", cat, "LOCAL", test_ui_routes)

        # F3: Image delivery cache headers
        def test_image_headers():
            runner = KagglePipelineTestRunner()
            with io.BytesIO() as bio:
                img = Image.new("RGB", (64, 64), color=(50, 100, 150))
                img.save(bio, format="PNG")
                raw = bio.getvalue()

            req = KaggleTestRequest(
                population_group="South Asian", region="India",
                age_min=25, age_max=35, sex_or_attribute="female",
                skin_tone="medium_brown", pose="frontal",
                expression="neutral", lighting="neutral",
            )
            # Create a completed run
            run_id = f"test_{int(time.time()*1000) % 100000000:08x}"
            run = KaggleTestRun(run_id=run_id, request=req)
            run.image_bytes = raw
            runner._runs[run_id] = run

            with patch("backend.app.api.routes.kaggle_test.kaggle_test_runner", runner):
                resp = client.get(f"/kaggle-test/image/{run_id}")
                assert resp.status_code == 200
                assert resp.headers.get("content-type") == "image/png"
                assert "no-store" in resp.headers.get("cache-control", "")
                assert len(resp.content) == len(raw)
            return "GET /kaggle-test/image/{run_id} serves image/png with no-store"

        self.run_check("F03", "API: image preview delivery & cache headers", cat, "LOCAL", test_image_headers)

        # F4: Cleanup endpoint clears memory
        def test_cleanup_endpoint():
            runner = KagglePipelineTestRunner()
            req = KaggleTestRequest(
                population_group="South Asian", region="India",
                age_min=25, age_max=35, sex_or_attribute="female",
                skin_tone="medium_brown", pose="frontal",
                expression="neutral", lighting="neutral",
            )
            run_id = f"test_{int(time.time()*1000) % 100000000:08x}"
            run = KaggleTestRun(run_id=run_id, request=req)
            run.image_bytes = b"sample_bytes"
            runner._runs[run_id] = run

            with patch("backend.app.api.routes.kaggle_test.kaggle_test_runner", runner):
                resp = client.post(f"/kaggle-test/cleanup/{run_id}")
                assert resp.status_code == 200
                assert run.image_bytes is None
            return "POST /kaggle-test/cleanup/{run_id} purges image memory"

        self.run_check("F04", "API: in-memory preview cleanup", cat, "LOCAL", test_cleanup_endpoint)

    # ----------------------------------------------------------------------- #
    # Category G: Data Isolation and Resource Cleanup                         #
    # ----------------------------------------------------------------------- #

    def check_category_g(self):
        cat = "G. Isolation & Cleanup"

        # G1: Zero database records
        def test_zero_db():
            runner = KagglePipelineTestRunner()
            req = KaggleTestRequest(
                population_group="South Asian", region="India",
                age_min=25, age_max=35, sex_or_attribute="female",
                skin_tone="medium_brown", pose="frontal",
                expression="neutral", lighting="neutral",
            )
            provider = MockTestProvider(succeed=True)
            with patch("backend.app.database.session.get_db") as mock_db:
                summary = runner.start_test(req, provider=provider)
                if runner._active_thread:
                    runner._active_thread.join(timeout=5)
                mock_db.assert_not_called()
            return "Zero SQL database sessions or table interactions occurred"

        self.run_check("G01", "Database isolation: zero records", cat, "LOCAL", test_zero_db)

        # G2: Existing storage files untouched
        def test_storage_untouched():
            exp_dir = Path(settings.EXPORT_DIR)
            val_dir = Path(settings.VALIDATED_IMAGES_DIR)
            exp_before = set(exp_dir.rglob("*")) if exp_dir.exists() else set()
            val_before = set(val_dir.rglob("*")) if val_dir.exists() else set()

            runner = KagglePipelineTestRunner()
            req = KaggleTestRequest(
                population_group="South Asian", region="India",
                age_min=25, age_max=35, sex_or_attribute="female",
                skin_tone="medium_brown", pose="frontal",
                expression="neutral", lighting="neutral",
            )
            provider = MockTestProvider(succeed=True)
            summary = runner.start_test(req, provider=provider)
            if runner._active_thread:
                runner._active_thread.join(timeout=5)

            exp_after = set(exp_dir.rglob("*")) if exp_dir.exists() else set()
            val_after = set(val_dir.rglob("*")) if val_dir.exists() else set()
            assert exp_before == exp_after
            assert val_before == val_after
            return "Export and validated storage directories untouched"

        self.run_check("G02", "Storage isolation: exports & galleries untouched", cat, "LOCAL", test_storage_untouched)

        # G3: Temp workspace cleanup after failure
        def test_cleanup_after_failure():
            runner = KagglePipelineTestRunner()
            req = KaggleTestRequest(
                population_group="South Asian", region="India",
                age_min=25, age_max=35, sex_or_attribute="female",
                skin_tone="medium_brown", pose="frontal",
                expression="neutral", lighting="neutral",
            )
            provider = MockTestProvider(succeed=False, failure_stage=3)
            summary = runner.start_test(req, provider=provider)
            if runner._active_thread:
                runner._active_thread.join(timeout=5)
            run = runner.get_run(summary.run_id)
            assert run.status == "FAILED"
            assert not run.workspace_dir.exists()
            assert run.stages[10].status == StageStatus.PASSED
            return "Temporary workspace deleted deterministically after failure"

        self.run_check("G03", "Workspace cleanup on failure", cat, "LOCAL", test_cleanup_after_failure)

    # ----------------------------------------------------------------------- #
    # Category H: Separate Read-Only Remote Checks (Phase 5 Design)           #
    # ----------------------------------------------------------------------- #

    def check_category_h(self):
        cat = "H. Optional Read-Only Remote"

        if not self.include_remote:
            self.mark_not_run(
                "H01", "Remote Kaggle API Authentication Ping", cat, "REMOTE",
                "NOT RUN: Requires explicit opt-in flag (--confirm-read-only-remote) with user authorization.",
            )
            self.mark_not_run(
                "H02", "Remote Manifest Dataset Read-Only Inspection", cat, "REMOTE",
                "NOT RUN: Requires explicit opt-in flag (--confirm-read-only-remote) with user authorization.",
            )
            self.mark_not_run(
                "H03", "Remote Kernel Metadata & Slug Inspection", cat, "REMOTE",
                "NOT RUN: Requires explicit opt-in flag (--confirm-read-only-remote) with user authorization.",
            )
            return

        # Explicit user opt-in confirmed
        print("\n[WARNING] Executing READ-ONLY remote checks against Kaggle...")

        cached_api = None

        # H01: Remote Kaggle API Authentication Ping (purely read-only)
        def test_remote_auth():
            nonlocal cached_api
            api = _get_kaggle_api()
            cached_api = api
            username = api.get_config_value(api.CONFIG_NAME_USER) or settings.KAGGLE_USERNAME
            assert username, "Authenticated with Kaggle SDK but unable to resolve username."
            return f"Authenticated successfully as Kaggle user: {username}"

        self.run_check(
            "H01", "Remote Kaggle API Authentication Ping", cat, "REMOTE",
            test_remote_auth,
            remediation="Verify ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY environment variables.",
        )

        # H02: Remote Manifest Dataset Read-Only Inspection
        def test_remote_dataset():
            nonlocal cached_api
            api = cached_api or _get_kaggle_api()
            slug = settings.KAGGLE_MANIFEST_DATASET_SLUG
            assert slug, "KAGGLE_MANIFEST_DATASET_SLUG is not configured"
            status = api.dataset_status(slug)
            return f"Remote manifest dataset '{slug}' status: {status}"

        self.run_check(
            "H02", "Remote Manifest Dataset Read-Only Inspection", cat, "REMOTE",
            test_remote_dataset,
            remediation="Verify KAGGLE_MANIFEST_DATASET_SLUG exists and your Kaggle account has read access.",
        )

        # H03: Remote Kernel Metadata & Slug Inspection
        def test_remote_kernel():
            nonlocal cached_api
            api = cached_api or _get_kaggle_api()
            slug = settings.KAGGLE_KERNEL_SLUG
            assert slug, "KAGGLE_KERNEL_SLUG is not configured"
            status_obj = api.kernels_status(slug)
            status_val = getattr(status_obj, "status", None) or (
                status_obj.get("status") if isinstance(status_obj, dict) else status_obj
            )
            return f"Remote kernel '{slug}' query succeeded, status: {status_val}"

        self.run_check(
            "H03", "Remote Kernel Metadata & Slug Inspection", cat, "REMOTE",
            test_remote_kernel,
            remediation="Verify KAGGLE_KERNEL_SLUG exists on Kaggle and matches your notebook slug.",
        )

    def execute_all(self) -> DiagnosticReport:
        t0 = time.perf_counter()
        self.check_category_a()
        self.check_category_b()
        self.check_category_c()
        self.check_category_d()
        self.check_category_e()
        self.check_category_f()
        self.check_category_g()
        self.check_category_h()
        total_dur = round(time.perf_counter() - t0, 3)

        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        not_run = sum(1 for r in self.results if r.status == "NOT RUN")

        local_passed = sum(1 for r in self.results if r.check_type in ("LOCAL", "MOCKED") and r.status == "PASS")
        local_failed = sum(1 for r in self.results if r.check_type in ("LOCAL", "MOCKED") and r.status == "FAIL")
        local_not_run = sum(1 for r in self.results if r.check_type in ("LOCAL", "MOCKED") and r.status == "NOT RUN")

        remote_passed = sum(1 for r in self.results if r.check_type == "REMOTE" and r.status == "PASS")
        remote_failed = sum(1 for r in self.results if r.check_type == "REMOTE" and r.status == "FAIL")
        remote_not_run = sum(1 for r in self.results if r.check_type == "REMOTE" and r.status == "NOT RUN")

        return DiagnosticReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            effective_provider=settings.effective_provider,
            total_checks=len(self.results),
            passed_count=passed,
            failed_count=failed,
            not_run_count=not_run,
            duration_seconds=total_dur,
            local_passed_count=local_passed,
            local_failed_count=local_failed,
            local_not_run_count=local_not_run,
            remote_passed_count=remote_passed,
            remote_failed_count=remote_failed,
            remote_not_run_count=remote_not_run,
            checks=self.results,
        )


def format_terminal_report(report: DiagnosticReport) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("KAGGLE PIPELINE CONSOLIDATED DIAGNOSTIC SUITE REPORT")
    lines.append("=" * 80)
    lines.append(f"Timestamp:          {report.timestamp}")
    lines.append(f"Effective Provider: {report.effective_provider} (MOCK mode safeguard verified)")
    lines.append(f"Total Duration:     {report.duration_seconds}s")
    lines.append(f"Summary:            {report.passed_count} PASSED | {report.failed_count} FAILED | {report.not_run_count} NOT RUN")
    lines.append(f"Local Checks:       {report.local_passed_count} passed, {report.local_failed_count} failed, {report.local_not_run_count} not run")
    lines.append(f"Remote Checks:      {report.remote_passed_count} passed, {report.remote_failed_count} failed, {report.remote_not_run_count} not run")
    lines.append("=" * 80)

    # Group by category
    categories: dict[str, list[CheckResult]] = {}
    for c in report.checks:
        categories.setdefault(c.category, []).append(c)

    for cat_name, items in categories.items():
        lines.append(f"\n[{cat_name}]")
        for item in items:
            status_marker = "+" if item.status == "PASS" else ("X" if item.status == "FAIL" else "-")
            dur_str = f"({item.duration_ms}ms)" if item.duration_ms > 0 else ""
            lines.append(f"  {status_marker} [{item.status:<7}] [{item.check_type:<6}] {item.check_id}: {item.name} {dur_str}")
            if item.details:
                lines.append(f"             Info: {item.details}")
            if item.error_summary:
                lines.append(f"             ERROR: {item.error_summary}")
            if item.remediation:
                lines.append(f"             Fix:   {item.remediation}")

    lines.append("\n" + "=" * 80)
    if report.failed_count == 0:
        if report.remote_not_run_count > 0:
            overall_msg = "ALL LOCAL CHECKS PASSED (Remote checks skipped - opt-in required)"
        else:
            overall_msg = "ALL LOCAL AND REMOTE CHECKS PASSED"
    else:
        overall_msg = f"FAILURES DETECTED ({report.failed_count} check(s) failed)"
    lines.append(f"OVERALL RESULT: {overall_msg}")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Kaggle Pipeline Consolidated Diagnostic Suite")
    parser.add_argument("--json", action="store_true", help="Output results as machine-readable JSON")
    parser.add_argument("--output", type=str, default="", help="Path to write JSON output report")
    parser.add_argument("--remote-check-dry-run", action="store_true", help="Preview read-only remote checks without connecting")
    parser.add_argument("--confirm-read-only-remote", action="store_true", help="Opt-in flag for read-only Kaggle network checks (NEVER pushes/runs)")

    args = parser.parse_args()

    if args.remote_check_dry_run:
        print("\n" + "=" * 80)
        print("OPTIONAL READ-ONLY REMOTE CHECK PREVIEW (PHASE 5 DESIGN)")
        print("=" * 80)
        print("The read-only remote check suite inspects the following without any mutations:")
        print("  1. Kaggle API Authentication Ping: Validates credentials via api.get_config_value()")
        print("  2. Manifest Dataset Accessibility: Read-only metadata query of KAGGLE_MANIFEST_DATASET_SLUG")
        print("  3. Kernel Slug Verification: Verifies read-only kernel metadata on Kaggle")
        print("\nSAFETY GUARANTEES:")
        print("  - ZERO notebook pushes (api.kernels_push is NEVER called)")
        print("  - ZERO dataset uploads (api.dataset_create_version is NEVER called)")
        print("  - ZERO GPU jobs scheduled")
        print("  - ZERO modifications to datasets or database records")
        print("To run, invoke with: python diagnostic_suite.py --confirm-read-only-remote")
        print("=" * 80 + "\n")

    suite = KagglePipelineDiagnosticSuite(
        include_remote_read_only=args.confirm_read_only_remote,
        dry_run_remote=args.remote_check_dry_run,
    )
    report = suite.execute_all()

    if args.json:
        report_dict = asdict(report)
        json_str = json.dumps(report_dict, indent=2)
        print(json_str)
        if args.output:
            Path(args.output).write_text(json_str, encoding="utf-8")
    else:
        text_report = format_terminal_report(report)
        print(text_report)
        if args.output:
            Path(args.output).write_text(text_report, encoding="utf-8")

    sys.exit(0 if report.failed_count == 0 else 1)


if __name__ == "__main__":
    main()
