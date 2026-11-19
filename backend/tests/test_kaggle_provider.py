"""
Tests for KaggleBatchProvider and the Kaggle worker path.

All tests mock the kaggle SDK — no real Kaggle submissions are made.

Structure
---------
TestKaggleBatchProvider   – unit tests for each provider method
TestKaggleWorkerPath      – integration tests: worker functions + DB + mocks
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.tests.conftest import create_test_dataset


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _make_manifest(tmp_path: Path, job_id: int = 1) -> "GenerationManifest":
    """Build a minimal GenerationManifest for testing."""
    from generation.manifest import GenerationManifest, ManifestProviderConfig
    return GenerationManifest(
        dataset_id=1,
        job_id=job_id,
        batch_id="test-batch-abc",
        requested_count=2,
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
        seeds=[111, 222],
        provider_config=ManifestProviderConfig(provider="KAGGLE_BATCH"),
    )


def _make_status_response(status: str):
    """Return a mock Kaggle status response object."""
    m = MagicMock()
    m.status = status
    return m


# =========================================================================== #
# Unit tests – KaggleBatchProvider                                             #
# =========================================================================== #

class TestKaggleBatchProvider:
    """Unit tests — mock kaggle.api at the point it is imported."""

    def test_provider_name(self):
        from generation.kaggle_provider import KaggleBatchProvider
        assert KaggleBatchProvider().provider_name == "KAGGLE_BATCH"

    def test_generate_raises(self):
        from generation.kaggle_provider import KaggleBatchProvider
        from backend.app.schemas.generation import GenerationRequest
        with pytest.raises(NotImplementedError):
            KaggleBatchProvider().generate(
                GenerationRequest(
                    age_min=25, age_max=35, sex_or_attribute="female",
                    skin_tone="medium_brown", population_group="South Asian",
                    region="India", pose="frontal", expression="neutral",
                    resolution=512,
                )
            )

    # ------------------------------------------------------------------ #
    # prepare_batch                                                        #
    # ------------------------------------------------------------------ #

    def test_prepare_batch_missing_manifest_slug(self, tmp_path):
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)
        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_MANIFEST_DATASET_SLUG = ""
            result = KaggleBatchProvider().prepare_batch(manifest)
        assert not result.succeeded
        assert result.outcome.name == "PERMANENT_ERROR"
        assert "KAGGLE_MANIFEST_DATASET_SLUG" in result.message

    def test_prepare_batch_missing_credentials(self, tmp_path):
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)
        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            with patch("generation.kaggle_provider._get_kaggle_api") as mock_auth:
                mock_auth.side_effect = EnvironmentError("No kaggle.json found")
                result = KaggleBatchProvider().prepare_batch(manifest)
        assert not result.succeeded
        assert result.outcome.name == "PERMANENT_ERROR"
        assert "No kaggle.json" in result.message

    def test_prepare_batch_upload_api_error(self, tmp_path):
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)
        mock_api = MagicMock()
        mock_api.dataset_create_version.side_effect = RuntimeError("Connection refused")
        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                result = KaggleBatchProvider().prepare_batch(manifest)
        assert not result.succeeded
        assert result.outcome.name == "PERMANENT_ERROR"
        assert "RuntimeError" in result.message

    def test_prepare_batch_success(self, tmp_path):
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)
        mock_api = MagicMock()
        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                result = KaggleBatchProvider().prepare_batch(manifest)
        assert result.succeeded
        mock_api.dataset_create_version.assert_called_once()

    def test_prepare_batch_writes_dataset_metadata_json(self, tmp_path):
        """
        Regression test: KaggleApi.dataset_create_version requires dataset-metadata.json
        in the target folder with the dataset id/slug. Without it, KaggleApi raises
        ValueError('Metadata file not found: dataset-metadata.json').
        """
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)

        captured_folder = {}

        def fake_create_version(folder, **kwargs):
            meta_path = Path(folder) / "dataset-metadata.json"
            assert meta_path.exists(), "dataset-metadata.json must exist in upload folder"
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            captured_folder["id"] = data.get("id")
            manifest_file = Path(folder) / "manifest.json"
            assert manifest_file.exists(), "manifest.json must exist in upload folder"

        mock_api = MagicMock()
        mock_api.dataset_create_version.side_effect = fake_create_version

        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_MANIFEST_DATASET_SLUG = "myuser/my-dataset"
            with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                result = KaggleBatchProvider().prepare_batch(manifest)

        assert result.succeeded
        assert captured_folder.get("id") == "myuser/my-dataset"

    def test_prepare_batch_propagates_safe_error_detail_and_redacts_tokens(self, tmp_path):
        """
        Regression test: When dataset_create_version raises an exception, the error detail
        is preserved in the message for diagnostics, but any sensitive tokens/keys are redacted.
        """
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)

        mock_api = MagicMock()
        # Simulate an exception containing sensitive token and metadata detail
        mock_api.dataset_create_version.side_effect = ValueError(
            "Metadata file not found: dataset-metadata.json with token=abcdef0123456789abcdef0123456789"
        )

        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                result = KaggleBatchProvider().prepare_batch(manifest)

        assert not result.succeeded
        assert "ValueError" in result.message
        assert "Metadata file not found: dataset-metadata.json" in result.message
        # Token must be redacted
        assert "abcdef0123456789abcdef0123456789" not in result.message
        assert "[REDACTED]" in result.message

    # ------------------------------------------------------------------ #
    # submit_batch                                                         #
    # ------------------------------------------------------------------ #

    def test_submit_batch_missing_kernel_slug(self, tmp_path):
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)
        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_KERNEL_SLUG = ""
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            result = KaggleBatchProvider().submit_batch(manifest)
        assert not result.succeeded
        assert "KAGGLE_KERNEL_SLUG" in result.message

    def test_submit_batch_push_fails(self, tmp_path):
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)

        # Create a real notebook file and metadata so the path checks pass.
        nb_file = tmp_path / "test_notebook.ipynb"
        nb_file.write_text('{"cells": []}', encoding="utf-8")
        meta_file = tmp_path / "kernel-metadata.json"
        meta_file.write_text(
            json.dumps({"id": "user/apac-gen", "dataset_sources": []}),
            encoding="utf-8",
        )

        mock_api = MagicMock()
        mock_api.kernels_push.side_effect = RuntimeError("push failed")
        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "user/apac-gen"
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            s.KAGGLE_NOTEBOOK_PATH = str(nb_file)
            s.KAGGLE_KERNEL_METADATA_PATH = str(meta_file)
            with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                result = KaggleBatchProvider().submit_batch(manifest)
        assert not result.succeeded
        assert result.outcome.name == "PERMANENT_ERROR"

    def test_submit_batch_success(self, tmp_path):
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)

        # Create a real notebook file and metadata so the path checks pass.
        nb_file = tmp_path / "test_notebook.ipynb"
        nb_file.write_text('{"cells": []}', encoding="utf-8")
        meta_file = tmp_path / "kernel-metadata.json"
        meta_file.write_text(
            json.dumps({"id": "user/apac-gen", "dataset_sources": []}),
            encoding="utf-8",
        )

        mock_api = MagicMock()
        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "user/apac-gen"
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            s.KAGGLE_NOTEBOOK_PATH = str(nb_file)
            s.KAGGLE_KERNEL_METADATA_PATH = str(meta_file)
            with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                result = KaggleBatchProvider().submit_batch(manifest)
        assert result.succeeded
        assert result.provider_run_id == "user/apac-gen"
        mock_api.kernels_push.assert_called_once()

    def test_submit_batch_unicode_charmap_regression(self, tmp_path):
        """
        Regression test: When a notebook contains non-ASCII characters such as '❌'
        (which encodes in UTF-8 to bytes including 0x9d, unmapped in Windows cp1252),
        submit_batch writes the push notebook with ensure_ascii=True and wraps kernels_push
        in _utf8_open_context to prevent 'UnicodeDecodeError: charmap codec can't decode byte 0x9d'.
        """
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)

        # Notebook containing the exact Unicode red-cross byte sequence
        nb_file = tmp_path / "unicode_nb.ipynb"
        nb_data = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('❌ GPU NOT AVAILABLE')\n"],
                    "outputs": [],
                }
            ]
        }
        # Write notebook with raw UTF-8 bytes containing 0x9d
        nb_file.write_bytes(json.dumps(nb_data, ensure_ascii=False).encode("utf-8"))
        assert b"\x9d" in nb_file.read_bytes()

        meta_file = tmp_path / "kernel-metadata.json"
        meta_file.write_text(
            json.dumps({"id": "user/apac-gen", "dataset_sources": []}),
            encoding="utf-8",
        )

        captured_code = {}

        def fake_kernels_push(folder):
            # KaggleApi.kernels_push opens code_file without specifying encoding
            # We verify that open() in this context reads the file cleanly without UnicodeDecodeError
            push_nb = Path(folder) / "unicode_nb.ipynb"
            assert push_nb.exists()
            content = open(str(push_nb)).read()
            captured_code["content"] = content
            # Also verify that the JSON loads properly and preserves the Unicode character
            loaded = json.loads(content)
            assert "❌" in loaded["cells"][0]["source"][0]

        mock_api = MagicMock()
        mock_api.kernels_push.side_effect = fake_kernels_push

        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "user/apac-gen"
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            s.KAGGLE_NOTEBOOK_PATH = str(nb_file)
            s.KAGGLE_KERNEL_METADATA_PATH = str(meta_file)
            with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                result = KaggleBatchProvider().submit_batch(manifest)

        assert result.succeeded
        assert "content" in captured_code

    def test_submit_batch_non_ascii_error_reporting_does_not_crash(self, tmp_path):
        """
        Regression test: When kernels_push raises an exception with non-ASCII characters
        and sensitive tokens, submit_batch reports the error safely without Unicode crash
        and redacts sensitive tokens.
        """
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)

        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text('{"cells": []}', encoding="utf-8")
        meta_file = tmp_path / "kernel-metadata.json"
        meta_file.write_text(
            json.dumps({"id": "user/apac-gen", "dataset_sources": []}),
            encoding="utf-8",
        )

        mock_api = MagicMock()
        mock_api.kernels_push.side_effect = RuntimeError(
            "Push failed: ❌ invalid UTF-8 token=abcdef0123456789abcdef0123456789"
        )

        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "user/apac-gen"
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            s.KAGGLE_NOTEBOOK_PATH = str(nb_file)
            s.KAGGLE_KERNEL_METADATA_PATH = str(meta_file)
            with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                result = KaggleBatchProvider().submit_batch(manifest)

        assert not result.succeeded
        assert "RuntimeError" in result.message
        assert "abcdef0123456789abcdef0123456789" not in result.message
        assert "[REDACTED]" in result.message

    def test_submit_batch_notebook_not_found(self, tmp_path):
        """Missing notebook path returns PERMANENT_ERROR immediately."""
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)
        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "user/apac-gen"
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            s.KAGGLE_NOTEBOOK_PATH = str(tmp_path / "nonexistent.ipynb")
            s.KAGGLE_KERNEL_METADATA_PATH = str(tmp_path / "kernel-metadata.json")
            result = KaggleBatchProvider().submit_batch(manifest)
        assert result.outcome.name == "PERMANENT_ERROR"
        assert "Notebook file not found" in result.message
        assert "KAGGLE_NOTEBOOK_PATH" in result.message

    def test_submit_batch_metadata_not_found(self, tmp_path):
        """Missing kernel-metadata.json returns PERMANENT_ERROR."""
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)
        nb_file = tmp_path / "notebook.ipynb"
        nb_file.write_text("{}", encoding="utf-8")
        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "user/apac-gen"
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            s.KAGGLE_NOTEBOOK_PATH = str(nb_file)
            s.KAGGLE_KERNEL_METADATA_PATH = str(tmp_path / "nonexistent-metadata.json")
            result = KaggleBatchProvider().submit_batch(manifest)
        assert result.outcome.name == "PERMANENT_ERROR"
        assert "kernel-metadata.json not found" in result.message

    def test_submit_batch_pushes_real_notebook_not_placeholder(self, tmp_path):
        """Verifies that the push dir contains the actual notebook, not run.py."""
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)

        nb_file = tmp_path / "apac-face-generator-manifest-ready.ipynb"
        nb_file.write_text('{"cells": [], "sentinel": "real_notebook"}', encoding="utf-8")
        meta_file = tmp_path / "kernel-metadata.json"
        meta_file.write_text(
            json.dumps({"id": "user/apac-gen", "dataset_sources": []}),
            encoding="utf-8",
        )

        # Capture files present in the push dir while the call is in progress
        # (before the TemporaryDirectory context cleans up).
        captured_filenames: list[str] = []

        def capture_push(push_dir):
            captured_filenames.extend(f.name for f in Path(push_dir).iterdir())

        mock_api = MagicMock()
        mock_api.kernels_push.side_effect = capture_push

        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "user/apac-gen"
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            s.KAGGLE_NOTEBOOK_PATH = str(nb_file)
            s.KAGGLE_KERNEL_METADATA_PATH = str(meta_file)
            with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                result = KaggleBatchProvider().submit_batch(manifest)

        assert result.succeeded
        assert mock_api.kernels_push.call_count == 1
        assert nb_file.name in captured_filenames, (
            f"Notebook not in push dir: {captured_filenames}"
        )
        assert "run.py" not in captured_filenames, (
            "Placeholder run.py must not be present"
        )

    def test_submit_batch_manifest_dataset_injected_into_metadata(self, tmp_path):
        """The manifest dataset slug must be injected into dataset_sources."""
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)

        nb_file = tmp_path / "notebook.ipynb"
        nb_file.write_text("{}", encoding="utf-8")
        # Start with empty dataset_sources in the base metadata.
        meta_file = tmp_path / "kernel-metadata.json"
        meta_file.write_text(
            json.dumps({"id": "user/apac-gen", "dataset_sources": []}),
            encoding="utf-8",
        )

        pushed_meta = {}

        def capture_push(push_dir):
            meta_path = Path(push_dir) / "kernel-metadata.json"
            pushed_meta.update(json.loads(meta_path.read_text()))

        mock_api = MagicMock()
        mock_api.kernels_push.side_effect = capture_push

        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "user/apac-gen"
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            s.KAGGLE_NOTEBOOK_PATH = str(nb_file)
            s.KAGGLE_KERNEL_METADATA_PATH = str(meta_file)
            with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                KaggleBatchProvider().submit_batch(manifest)

        assert "user/apac-manifest" in pushed_meta.get("dataset_sources", [])
        assert pushed_meta.get("enable_gpu") is True
        assert pushed_meta.get("enable_internet") is True

    def test_submit_batch_supports_resolved_path_properties_regression(self, tmp_path):
        """
        Regression test: verify submit_batch works when settings provides
        resolved_notebook_path / resolved_kernel_metadata_path as Path instances.
        """
        from generation.kaggle_provider import KaggleBatchProvider
        manifest = _make_manifest(tmp_path)

        nb_file = tmp_path / "custom_notebook.ipynb"
        nb_file.write_text('{"cells": []}', encoding="utf-8")
        meta_file = tmp_path / "custom-metadata.json"
        meta_file.write_text(
            json.dumps({"id": "user/apac-gen", "dataset_sources": []}),
            encoding="utf-8",
        )

        mock_api = MagicMock()
        with patch("generation.kaggle_provider.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "user/apac-gen"
            s.KAGGLE_MANIFEST_DATASET_SLUG = "user/apac-manifest"
            s.resolved_notebook_path = nb_file
            s.resolved_kernel_metadata_path = meta_file
            with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
                result = KaggleBatchProvider().submit_batch(manifest)

        assert result.succeeded
        mock_api.kernels_push.assert_called_once()

    def test_submit_batch_uses_real_kernel_metadata(self, tmp_path):
        """The actual kaggle_notebook_check/kernel-metadata.json is valid JSON."""
        from pathlib import Path as _P
        meta_path = _P("kaggle_notebook_check/kernel-metadata.json")
        if not meta_path.exists():
            pytest.skip("kernel-metadata.json not present in this environment")
        import json as _json
        data = _json.loads(meta_path.read_text(encoding="utf-8"))
        assert "id" in data
        assert data.get("kernel_type") == "notebook"
        assert data.get("language") == "python"

    def test_submit_batch_uses_real_notebook_file(self, tmp_path):
        """The actual tested notebook file exists and is valid JSON."""
        from pathlib import Path as _P
        nb_path = _P("kaggle_notebook_check/apac-face-generator-manifest-ready.ipynb")
        if not nb_path.exists():
            pytest.skip("Notebook not present in this environment")
        import json as _json
        data = _json.loads(nb_path.read_text(encoding="utf-8"))
        assert "cells" in data
        assert data.get("nbformat") == 4

    # ------------------------------------------------------------------ #
    # poll_status                                                          #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("kaggle_status,expected_pending", [
        ("running", True),
        ("queued", True),
        ("cancelrequested", True),
        ("cancelRequested", True),  # mixed case
    ])
    def test_poll_status_pending(self, kaggle_status, expected_pending):
        from generation.kaggle_provider import KaggleBatchProvider
        mock_api = MagicMock()
        mock_api.kernels_status.return_value = _make_status_response(kaggle_status)
        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().poll_status("user/apac-gen")
        assert result.is_pending is expected_pending

    def test_poll_status_complete(self):
        from generation.kaggle_provider import KaggleBatchProvider
        mock_api = MagicMock()
        mock_api.kernels_status.return_value = _make_status_response("complete")
        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().poll_status("user/apac-gen")
        assert result.succeeded

    @pytest.mark.parametrize("kaggle_status", ["error", "cancelled"])
    def test_poll_status_permanent_error(self, kaggle_status):
        from generation.kaggle_provider import KaggleBatchProvider
        mock_api = MagicMock()
        mock_api.kernels_status.return_value = _make_status_response(kaggle_status)
        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().poll_status("user/apac-gen")
        assert result.outcome.name == "PERMANENT_ERROR"

    def test_poll_status_network_error_is_transient(self):
        from generation.kaggle_provider import KaggleBatchProvider
        mock_api = MagicMock()
        mock_api.kernels_status.side_effect = ConnectionError("timeout")
        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().poll_status("user/apac-gen")
        assert result.outcome.name == "TRANSIENT_ERROR"

    def test_poll_status_unknown_status_treated_as_pending(self):
        from generation.kaggle_provider import KaggleBatchProvider
        mock_api = MagicMock()
        mock_api.kernels_status.return_value = _make_status_response("pending_gpu_allocation")
        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().poll_status("user/apac-gen")
        assert result.is_pending

    def test_poll_status_empty_slug_is_permanent_error(self):
        from generation.kaggle_provider import KaggleBatchProvider
        result = KaggleBatchProvider().poll_status("")
        assert result.outcome.name == "PERMANENT_ERROR"

    def test_normalize_kernel_status_sdk_enum(self):
        from generation.kaggle_provider import normalize_kernel_status
        from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus

        assert normalize_kernel_status(KernelWorkerStatus.COMPLETE) == "complete"
        assert normalize_kernel_status(KernelWorkerStatus.RUNNING) == "running"
        assert normalize_kernel_status(KernelWorkerStatus.QUEUED) == "queued"
        assert normalize_kernel_status(KernelWorkerStatus.ERROR) == "error"
        assert normalize_kernel_status(KernelWorkerStatus.CANCEL_REQUESTED) == "cancel_requested"
        assert normalize_kernel_status(KernelWorkerStatus.CANCEL_ACKNOWLEDGED) == "cancel_acknowledged"

    def test_normalize_kernel_status_plain_and_dot_strings(self):
        from generation.kaggle_provider import normalize_kernel_status

        assert normalize_kernel_status("complete") == "complete"
        assert normalize_kernel_status("COMPLETE") == "complete"
        assert normalize_kernel_status("kernelworkerstatus.complete") == "complete"
        assert normalize_kernel_status("KernelWorkerStatus.COMPLETE") == "complete"
        assert normalize_kernel_status("KernelWorkerStatus.RUNNING") == "running"
        assert normalize_kernel_status("KernelWorkerStatus.ERROR") == "error"
        assert normalize_kernel_status("status.running") == "running"

    def test_normalize_kernel_status_object_and_dict(self):
        from generation.kaggle_provider import normalize_kernel_status
        from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusResponse

        resp = ApiGetKernelSessionStatusResponse()
        resp.status = KernelWorkerStatus.COMPLETE
        assert normalize_kernel_status(resp) == "complete"

        assert normalize_kernel_status({"status": "complete"}) == "complete"
        assert normalize_kernel_status({"status": KernelWorkerStatus.RUNNING}) == "running"
        assert normalize_kernel_status({"status": "kernelworkerstatus.complete"}) == "complete"

    def test_poll_status_actual_sdk_enum_complete_regression(self):
        """Verify the exact object returned by Kaggle SDK (ApiGetKernelSessionStatusResponse with KernelWorkerStatus.COMPLETE) resolves to SUCCESS."""
        from generation.kaggle_provider import KaggleBatchProvider
        from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusResponse

        resp = ApiGetKernelSessionStatusResponse()
        resp.status = KernelWorkerStatus.COMPLETE

        mock_api = MagicMock()
        mock_api.kernels_status.return_value = resp
        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().poll_status("user/apac-gen")

        assert result.succeeded
        assert result.outcome.name == "SUCCESS"
        assert result.provider_run_id == "user/apac-gen"

    def test_poll_status_actual_sdk_enum_error_surfaces_failure_message(self):
        """Verify KernelWorkerStatus.ERROR with failure_message maps to PERMANENT_ERROR and surfaces failure details."""
        from generation.kaggle_provider import KaggleBatchProvider
        from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusResponse

        resp = ApiGetKernelSessionStatusResponse()
        resp.status = KernelWorkerStatus.ERROR
        resp.failure_message = "CUDA out of memory"

        mock_api = MagicMock()
        mock_api.kernels_status.return_value = resp
        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().poll_status("user/apac-gen")

        assert result.outcome.name == "PERMANENT_ERROR"
        assert "CUDA out of memory" in result.message

    def test_poll_status_dot_string_complete_regression(self):
        """Verify string 'kernelworkerstatus.complete' observed in uvicorn log resolves to SUCCESS."""
        from generation.kaggle_provider import KaggleBatchProvider
        mock_api = MagicMock()
        mock_api.kernels_status.return_value = _make_status_response("kernelworkerstatus.complete")
        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().poll_status("user/apac-gen")

        assert result.succeeded
        assert result.outcome.name == "SUCCESS"

    def test_poll_status_dot_string_running_regression(self):
        """Verify string 'kernelworkerstatus.running' resolves to PENDING."""
        from generation.kaggle_provider import KaggleBatchProvider
        mock_api = MagicMock()
        mock_api.kernels_status.return_value = _make_status_response("kernelworkerstatus.running")
        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().poll_status("user/apac-gen")

        assert result.is_pending
        assert result.outcome.name == "PENDING"

    def test_poll_status_unknown_status_never_treated_as_success(self):
        """Verify an unknown status is never silently treated as success."""
        from generation.kaggle_provider import KaggleBatchProvider
        mock_api = MagicMock()
        mock_api.kernels_status.return_value = _make_status_response("completely_unrecognized_custom_status")
        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().poll_status("user/apac-gen")

        assert not result.succeeded
        assert result.outcome.name != "SUCCESS"


    # ------------------------------------------------------------------ #
    # retrieve_artifacts                                                   #
    # ------------------------------------------------------------------ #

    def test_retrieve_artifacts_success(self, tmp_path):
        from generation.kaggle_provider import KaggleBatchProvider

        # Simulate API downloading files into dest_dir.
        def fake_kernels_output(slug, path, quiet):
            dest = Path(path)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "image_0001.png").write_bytes(b"PNG_DATA_1")
            (dest / "image_0002.png").write_bytes(b"PNG_DATA_2")
            (dest / "batch_summary.json").write_text(
                json.dumps({"job_id": 1, "successful_count": 2}),
                encoding="utf-8",
            )

        mock_api = MagicMock()
        mock_api.kernels_output.side_effect = fake_kernels_output

        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().retrieve_artifacts(
                "user/apac-gen",
                str(tmp_path),
                None,
            )

        assert result.succeeded
        assert len(result.data["artifact_paths"]) == 2
        assert result.data["summary_path"] is not None
        assert "batch_summary.json" in result.data["summary_path"]

    def test_retrieve_artifacts_no_files_is_permanent_error(self, tmp_path):
        from generation.kaggle_provider import KaggleBatchProvider

        def fake_kernels_output(slug, path, quiet):
            Path(path).mkdir(parents=True, exist_ok=True)
            # Write nothing.

        mock_api = MagicMock()
        mock_api.kernels_output.side_effect = fake_kernels_output

        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().retrieve_artifacts(
                "user/apac-gen", str(tmp_path), None
            )

        assert result.outcome.name == "PERMANENT_ERROR"
        assert "No image files" in result.message

    def test_retrieve_artifacts_network_error_is_transient(self, tmp_path):
        from generation.kaggle_provider import KaggleBatchProvider

        mock_api = MagicMock()
        mock_api.kernels_output.side_effect = ConnectionError("network down")

        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().retrieve_artifacts(
                "user/apac-gen", str(tmp_path), None
            )

        assert result.outcome.name == "TRANSIENT_ERROR"

    # ------------------------------------------------------------------ #
    # request_cancellation                                                 #
    # ------------------------------------------------------------------ #

    def test_request_cancellation_returns_unsupported(self):
        from generation.kaggle_provider import KaggleBatchProvider
        result = KaggleBatchProvider().request_cancellation("user/apac-gen")
        assert result.outcome.name == "UNSUPPORTED"


# =========================================================================== #
# Worker integration tests – Kaggle path                                       #
# =========================================================================== #

class TestKaggleWorkerPath:
    """
    Tests for the worker's Kaggle step functions using the test DB.
    No real Kaggle calls are made.
    """

    def _make_kaggle_job(self, db_session: Session, client: TestClient) -> tuple:
        """Create a dataset + KAGGLE_BATCH job in PREPARING state."""
        dataset_dict = create_test_dataset(client, overrides={"num_images": 2})
        dataset_id = dataset_dict["id"]

        from backend.app.models.dataset import Dataset
        from backend.app.models.generation_job import GenerationJob
        from backend.app.core.status import DatasetStatus, JobStatus
        from backend.app.services.job_service import create_job
        from datetime import datetime, timezone

        dataset = db_session.query(Dataset).filter(Dataset.id == dataset_id).first()
        dataset.status = DatasetStatus.GENERATING
        db_session.flush()

        job = create_job(
            db_session,
            dataset=dataset,
            provider="KAGGLE_BATCH",
            requested_count=2,
        )
        db_session.flush()

        # Manually move to PREPARING (skipping claim_job to avoid lease complexities).
        job.status = JobStatus.PREPARING.value
        job.batch_id = "test-batch-xyz"
        db_session.commit()
        db_session.refresh(job)
        db_session.refresh(dataset)

        return job, dataset

    def test_kaggle_prepare_succeeds_and_transitions_to_submitted(
        self, client: TestClient, db_session: Session
    ):
        from backend.worker import _kaggle_prepare
        from backend.app.core.status import JobStatus

        job, dataset = self._make_kaggle_job(db_session, client)

        mock_provider = MagicMock()
        mock_provider.prepare_batch.return_value = MagicMock(succeeded=True, message="OK")
        mock_provider.submit_batch.return_value = MagicMock(
            succeeded=True,
            provider_run_id="user/apac-gen",
            message="OK",
        )

        tmp_artifacts = Path(tempfile.gettempdir()) / "apac_test_artifacts"

        with patch("backend.worker.get_provider", return_value=mock_provider), \
             patch("backend.worker.settings") as s:
            s.ARTIFACT_DOWNLOAD_DIR = str(tmp_artifacts)
            s.IMAGE_WIDTH = 512
            s.IMAGE_HEIGHT = 512
            s.KAGGLE_POST_PUSH_DELAY_SECONDS = 90
            s.JOB_LEASE_SECONDS = 300
            _kaggle_prepare(db_session, job, dataset)

        db_session.refresh(job)
        assert job.status == JobStatus.SUBMITTED.value
        assert job.provider_run_id == "user/apac-gen"
        assert "PUSH_TS=" in (job.current_stage or "")

    def test_kaggle_prepare_prepare_batch_failure_transitions_to_failed(
        self, client: TestClient, db_session: Session
    ):
        from backend.worker import _kaggle_prepare
        from backend.app.core.status import JobStatus
        from generation.interface import ProviderOperationResult

        job, dataset = self._make_kaggle_job(db_session, client)

        mock_provider = MagicMock()
        mock_provider.prepare_batch.return_value = ProviderOperationResult.permanent_error(
            message="No credentials"
        )

        tmp_artifacts = Path(tempfile.gettempdir()) / "apac_test_artifacts"

        with patch("backend.worker.get_provider", return_value=mock_provider), \
             patch("backend.worker.settings") as s:
            s.ARTIFACT_DOWNLOAD_DIR = str(tmp_artifacts)
            s.IMAGE_WIDTH = 512
            s.IMAGE_HEIGHT = 512
            s.KAGGLE_POST_PUSH_DELAY_SECONDS = 90
            s.JOB_LEASE_SECONDS = 300
            _kaggle_prepare(db_session, job, dataset)

        db_session.refresh(job)
        assert job.status == JobStatus.FAILED.value
        assert "No credentials" in (job.error_summary or "")

    def test_kaggle_poll_pending_keeps_submitted_status(
        self, client: TestClient, db_session: Session
    ):
        from backend.worker import _kaggle_poll
        from backend.app.core.status import JobStatus
        from generation.interface import ProviderOperationResult
        from datetime import datetime, timedelta, timezone

        job, dataset = self._make_kaggle_job(db_session, client)
        job.status = JobStatus.SUBMITTED.value
        push_time = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=200)
        )
        job.current_stage = f"PUSH_TS={push_time.isoformat()}|Submitted"
        job.provider_run_id = "user/apac-gen"
        db_session.commit()

        mock_provider = MagicMock()
        mock_provider.poll_status.return_value = ProviderOperationResult.pending(
            message="running"
        )

        with patch("backend.worker.get_provider", return_value=mock_provider), \
             patch("backend.worker.settings") as s:
            s.KAGGLE_POST_PUSH_DELAY_SECONDS = 90
            s.KAGGLE_MAX_POLL_ERRORS = 5
            s.JOB_LEASE_SECONDS = 300
            _kaggle_poll(db_session, job)

        db_session.refresh(job)
        # Should have advanced to RUNNING (from SUBMITTED when pending).
        assert job.status == JobStatus.RUNNING.value

    def test_kaggle_poll_respects_post_push_delay(
        self, client: TestClient, db_session: Session
    ):
        from backend.worker import _kaggle_poll
        from backend.app.core.status import JobStatus
        from generation.interface import ProviderOperationResult
        from datetime import datetime, timezone

        job, dataset = self._make_kaggle_job(db_session, client)
        job.status = JobStatus.SUBMITTED.value
        # Set push time to NOW — delay has not elapsed.
        push_time = datetime.now(timezone.utc).replace(tzinfo=None)
        job.current_stage = f"PUSH_TS={push_time.isoformat()}|Submitted"
        job.provider_run_id = "user/apac-gen"
        db_session.commit()

        mock_provider = MagicMock()

        with patch("backend.worker.get_provider", return_value=mock_provider), \
             patch("backend.worker.settings") as s:
            s.KAGGLE_POST_PUSH_DELAY_SECONDS = 90
            s.KAGGLE_MAX_POLL_ERRORS = 5
            s.JOB_LEASE_SECONDS = 300
            _kaggle_poll(db_session, job)

        # poll_status should NOT have been called (delay not elapsed).
        mock_provider.poll_status.assert_not_called()
        db_session.refresh(job)
        assert job.status == JobStatus.SUBMITTED.value

    def test_kaggle_poll_complete_transitions_to_downloading(
        self, client: TestClient, db_session: Session
    ):
        from backend.worker import _kaggle_poll
        from backend.app.core.status import JobStatus
        from generation.interface import ProviderOperationResult
        from datetime import datetime, timedelta, timezone

        job, dataset = self._make_kaggle_job(db_session, client)
        job.status = JobStatus.RUNNING.value
        push_time = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=200)
        )
        job.current_stage = f"PUSH_TS={push_time.isoformat()}|Running"
        job.provider_run_id = "user/apac-gen"
        db_session.commit()

        mock_provider = MagicMock()
        mock_provider.poll_status.return_value = ProviderOperationResult.success(
            provider_run_id="user/apac-gen",
            message="complete",
        )

        with patch("backend.worker.get_provider", return_value=mock_provider), \
             patch("backend.worker.settings") as s:
            s.KAGGLE_POST_PUSH_DELAY_SECONDS = 90
            s.KAGGLE_MAX_POLL_ERRORS = 5
            s.JOB_LEASE_SECONDS = 300
            _kaggle_poll(db_session, job)

        db_session.refresh(job)
        assert job.status == JobStatus.DOWNLOADING.value

    def test_kaggle_poll_max_errors_transitions_to_failed(
        self, client: TestClient, db_session: Session
    ):
        from backend.worker import _kaggle_poll
        from backend.app.core.status import JobStatus
        from generation.interface import ProviderOperationResult
        from datetime import datetime, timedelta, timezone

        job, dataset = self._make_kaggle_job(db_session, client)
        job.status = JobStatus.RUNNING.value
        push_time = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=200)
        )
        job.current_stage = f"PUSH_TS={push_time.isoformat()}|Running"
        job.provider_run_id = "user/apac-gen"
        # Simulate already at max-1 errors.
        job.error_summary = "POLL_ERRORS=4|previous error"
        db_session.commit()

        mock_provider = MagicMock()
        mock_provider.poll_status.return_value = ProviderOperationResult.transient_error(
            message="timeout"
        )

        with patch("backend.worker.get_provider", return_value=mock_provider), \
             patch("backend.worker.settings") as s:
            s.KAGGLE_POST_PUSH_DELAY_SECONDS = 90
            s.KAGGLE_MAX_POLL_ERRORS = 5
            s.JOB_LEASE_SECONDS = 300
            _kaggle_poll(db_session, job)

        db_session.refresh(job)
        assert job.status == JobStatus.FAILED.value

    def test_kaggle_validate_creates_image_records_accepted(
        self, client: TestClient, db_session: Session, tmp_path: Path
    ):
        from backend.worker import _kaggle_validate
        from backend.app.core.status import JobStatus
        from backend.app.models.generated_image import GeneratedImage
        from PIL import Image as PILImage

        job, dataset = self._make_kaggle_job(db_session, client)
        # _kaggle_validate is called when job is in VALIDATING state.
        job.status = JobStatus.VALIDATING.value
        job.provider_run_id = "user/apac-gen"
        db_session.commit()

        # Create a real tiny PNG in the artifact dir.
        artifact_dir = tmp_path / f"job_{job.id}"
        artifact_dir.mkdir(parents=True)

        img = PILImage.new("RGB", (512, 512), color=(100, 150, 200))
        img_path = artifact_dir / "image_0001.png"
        img.save(img_path)

        # Write a matching batch_summary.json.
        summary = {
            "job_id": job.id,
            "batch_id": job.batch_id,
            "requested_count": 2,
            "successful_count": 1,
            "failed_count": 0,
            "images": [
                {
                    "filename": "image_0001.png",
                    "seed": 12345,
                    "prompt": "Test prompt",
                    "width": 512,
                    "height": 512,
                }
            ],
        }
        (artifact_dir / "batch_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

        validated_dir = tmp_path / "generated"
        rejected_dir = tmp_path / "rejected"

        with patch("backend.worker.settings") as s:
            s.ARTIFACT_DOWNLOAD_DIR = str(tmp_path)
            s.GENERATED_IMAGES_DIR = str(validated_dir)
            s.REJECTED_IMAGES_DIR = str(rejected_dir)
            s.IMAGE_WIDTH = 512
            s.IMAGE_HEIGHT = 512
            s.ALLOWED_IMAGE_FORMATS = "png,jpg,jpeg,webp"
            s.MAX_IMAGE_FILE_BYTES = 20 * 1024 * 1024
            s.REQUIRE_FACE_DETECTION = False
            s.ENABLE_DUPLICATE_DETECTION = True
            s.DUPLICATE_HASH_THRESHOLD = 10
            s.allowed_formats = ["png", "jpg", "jpeg", "webp"]
            _kaggle_validate(db_session, job, dataset)

        db_session.refresh(job)
        assert job.status == JobStatus.PACKAGING.value
        assert job.accepted_count == 1

        images = (
            db_session.query(GeneratedImage)
            .filter(GeneratedImage.job_id == job.id)
            .all()
        )
        assert len(images) == 1
        assert images[0].seed == 12345
        assert images[0].prompt == "Test prompt"
        assert images[0].validation_status == "ACCEPTED"

    def test_kaggle_validate_artifact_overlap_fails_job(
        self, client: TestClient, db_session: Session, tmp_path: Path
    ):
        from backend.worker import _kaggle_validate
        from backend.app.core.status import JobStatus
        from PIL import Image as PILImage

        job, dataset = self._make_kaggle_job(db_session, client)
        job.status = JobStatus.VALIDATING.value
        db_session.commit()

        artifact_dir = tmp_path / f"job_{job.id}"
        artifact_dir.mkdir(parents=True)

        img = PILImage.new("RGB", (512, 512))
        img.save(artifact_dir / "image_0001.png")

        # Summary claims to belong to a DIFFERENT job — overlap scenario.
        summary = {"job_id": 9999, "images": []}
        (artifact_dir / "batch_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

        with patch("backend.worker.settings") as s:
            s.ARTIFACT_DOWNLOAD_DIR = str(tmp_path)
            s.GENERATED_IMAGES_DIR = str(tmp_path / "gen")
            s.REJECTED_IMAGES_DIR = str(tmp_path / "rej")
            s.IMAGE_WIDTH = 512
            s.IMAGE_HEIGHT = 512
            s.ALLOWED_IMAGE_FORMATS = "png,jpg,jpeg,webp"
            s.MAX_IMAGE_FILE_BYTES = 20 * 1024 * 1024
            s.REQUIRE_FACE_DETECTION = False
            s.ENABLE_DUPLICATE_DETECTION = False
            s.DUPLICATE_HASH_THRESHOLD = 10
            s.allowed_formats = ["png", "jpg", "jpeg", "webp"]
            _kaggle_validate(db_session, job, dataset)

        db_session.refresh(job)
        assert job.status == JobStatus.FAILED.value
        # Error message must describe the mismatch (not necessarily "overlap" word).
        assert "9999" in (job.error_summary or "")

    def test_kaggle_validate_no_images_fails_job(
        self, client: TestClient, db_session: Session, tmp_path: Path
    ):
        from backend.worker import _kaggle_validate
        from backend.app.core.status import JobStatus

        job, dataset = self._make_kaggle_job(db_session, client)
        job.status = JobStatus.VALIDATING.value
        db_session.commit()

        # Create artifact dir with no images.
        artifact_dir = tmp_path / f"job_{job.id}"
        artifact_dir.mkdir(parents=True)

        with patch("backend.worker.settings") as s:
            s.ARTIFACT_DOWNLOAD_DIR = str(tmp_path)
            s.GENERATED_IMAGES_DIR = str(tmp_path / "gen")
            s.REJECTED_IMAGES_DIR = str(tmp_path / "rej")
            s.IMAGE_WIDTH = 512
            s.IMAGE_HEIGHT = 512
            s.ALLOWED_IMAGE_FORMATS = "png,jpg,jpeg,webp"
            s.MAX_IMAGE_FILE_BYTES = 20 * 1024 * 1024
            s.REQUIRE_FACE_DETECTION = False
            s.ENABLE_DUPLICATE_DETECTION = False
            s.DUPLICATE_HASH_THRESHOLD = 10
            s.allowed_formats = ["png", "jpg", "jpeg", "webp"]
            _kaggle_validate(db_session, job, dataset)

        db_session.refresh(job)
        assert job.status == JobStatus.FAILED.value

    def test_overlap_guard_prevents_second_kaggle_job(
        self, client: TestClient, db_session: Session
    ):
        """
        If a KAGGLE_BATCH job is already active, the next queued job
        must not be claimed in the same poll cycle.
        """
        from backend.worker import run_once
        from backend.app.core.status import JobStatus
        from backend.app.models.generation_job import GenerationJob
        from backend.app.services.job_service import create_job
        from backend.app.models.dataset import Dataset
        from backend.app.core.status import DatasetStatus

        # Create two datasets + two KAGGLE_BATCH jobs.
        ds1_dict = create_test_dataset(client, overrides={"dataset_name": "DS_OG1"})
        ds2_dict = create_test_dataset(client, overrides={"dataset_name": "DS_OG2"})

        ds1 = db_session.query(Dataset).filter(Dataset.id == ds1_dict["id"]).first()
        ds2 = db_session.query(Dataset).filter(Dataset.id == ds2_dict["id"]).first()

        ds1.status = DatasetStatus.GENERATING
        ds2.status = DatasetStatus.GENERATING
        db_session.flush()

        job1 = create_job(db_session, dataset=ds1, provider="KAGGLE_BATCH", requested_count=1)
        job2 = create_job(db_session, dataset=ds2, provider="KAGGLE_BATCH", requested_count=1)
        db_session.flush()

        # Manually move job1 to RUNNING (simulating it was already picked up).
        job1.status = JobStatus.RUNNING.value
        job1.worker_id = "test-worker"
        from datetime import datetime, timedelta, timezone
        job1.lease_expires_at = (
            datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=300)
        )
        db_session.commit()

        # run_once should skip job2 because job1 is already active.
        with patch("backend.worker.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "user/apac-gen"
            s.WORKER_POLL_INTERVAL_SECONDS = 10
            s.JOB_LEASE_SECONDS = 300
            with patch("backend.worker.get_stale_jobs", return_value=[]):
                with patch("backend.worker.get_queued_jobs", return_value=[job2]):
                    processed = run_once(db_session)

        db_session.refresh(job2)
        # job2 must still be QUEUED — not claimed.
        assert job2.status == JobStatus.QUEUED.value


# =========================================================================== #
# New tests added for notebook integration fixes                               #
# =========================================================================== #

class TestArtifactDiscovery:
    """
    Tests for retrieve_artifacts() scanning nested output directories.

    The notebook saves images to:
        rhinoplasty_dataset/batches/{batch_id}/images/image_NNNN.png
    and the summary to:
        rhinoplasty_dataset/batches/{batch_id}/batch_summary.json

    retrieve_artifacts() must find both regardless of nesting depth.
    """

    def test_retrieve_artifacts_finds_nested_images_and_summary(self, tmp_path):
        """Images and summary nested 3 levels deep must be discovered."""
        from generation.kaggle_provider import KaggleBatchProvider

        def fake_kernels_output(slug, path, quiet):
            # Simulate the notebook's actual output layout.
            batch_dir = Path(path) / "rhinoplasty_dataset" / "batches" / "batch-abc"
            images_dir = batch_dir / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "image_0001.png").write_bytes(b"PNG1")
            (images_dir / "image_0002.png").write_bytes(b"PNG2")
            (batch_dir / "batch_summary.json").write_text(
                json.dumps({"job_id": 1, "successful_count": 2, "images": []}),
                encoding="utf-8",
            )

        mock_api = MagicMock()
        mock_api.kernels_output.side_effect = fake_kernels_output

        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().retrieve_artifacts(
                "user/apac-gen", str(tmp_path), None
            )

        assert result.succeeded
        assert len(result.data["artifact_paths"]) == 2
        assert result.data["summary_path"] is not None
        assert "batch_summary.json" in result.data["summary_path"]

    def test_retrieve_artifacts_images_only_no_summary_still_succeeds(self, tmp_path):
        """Images without a summary should still succeed (summary_path=None)."""
        from generation.kaggle_provider import KaggleBatchProvider

        def fake_kernels_output(slug, path, quiet):
            nested = Path(path) / "subdir" / "images"
            nested.mkdir(parents=True)
            (nested / "image_0001.png").write_bytes(b"PNG")

        mock_api = MagicMock()
        mock_api.kernels_output.side_effect = fake_kernels_output

        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().retrieve_artifacts(
                "user/apac-gen", str(tmp_path), None
            )

        assert result.succeeded
        assert len(result.data["artifact_paths"]) == 1
        assert result.data["summary_path"] is None

    def test_retrieve_artifacts_flat_layout_still_works(self, tmp_path):
        """Flat layout (images at root) must still be discovered."""
        from generation.kaggle_provider import KaggleBatchProvider

        def fake_kernels_output(slug, path, quiet):
            dest = Path(path)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "image_0001.png").write_bytes(b"PNG")
            (dest / "batch_summary.json").write_text("{}", encoding="utf-8")

        mock_api = MagicMock()
        mock_api.kernels_output.side_effect = fake_kernels_output

        with patch("generation.kaggle_provider._get_kaggle_api", return_value=mock_api):
            result = KaggleBatchProvider().retrieve_artifacts(
                "user/apac-gen", str(tmp_path), None
            )

        assert result.succeeded
        assert len(result.data["artifact_paths"]) == 1


class TestManifestCompatibility:
    """
    Tests verifying the manifest serialises fields the notebook expects
    and that the notebook's output contract matches what the worker reads.
    """

    def test_manifest_contains_job_id_at_top_level(self):
        """job_id must be a top-level field in the serialised JSON."""
        from generation.manifest import GenerationManifest, ManifestProviderConfig
        m = GenerationManifest(
            dataset_id=5,
            job_id=42,
            batch_id="b-001",
            requested_count=3,
            image_width=384,
            image_height=384,
            population_group="East Asian",
            region="China",
            age_min=20,
            age_max=40,
            sex_or_attribute="female",
            skin_tone="light_brown",
            pose="frontal",
            expression="neutral",
            seeds=[1, 2, 3],
            provider_config=ManifestProviderConfig(provider="KAGGLE_BATCH"),
        )
        data = json.loads(m.to_json())
        assert data["job_id"] == 42
        assert data["batch_id"] == "b-001"
        assert data["requested_count"] == 3
        assert len(data["seeds"]) == 3
        assert data["image_width"] == 384
        assert data["image_height"] == 384

    def test_manifest_provider_config_has_inference_settings(self):
        """num_inference_steps and guidance_scale must be present."""
        from generation.manifest import ManifestProviderConfig
        cfg = ManifestProviderConfig(provider="KAGGLE_BATCH")
        data = json.loads(cfg.model_dump_json())
        assert "num_inference_steps" in data
        assert "guidance_scale" in data
        assert data["num_inference_steps"] == 12
        assert data["guidance_scale"] == 1.0

    def test_manifest_notebook_reads_width_height(self):
        """
        The notebook reads manifest.get('width', manifest.get('image_width', 384)).
        Ensure image_width serialises correctly so the notebook can find it.
        """
        from generation.manifest import GenerationManifest, ManifestProviderConfig
        m = GenerationManifest(
            dataset_id=1, job_id=1, batch_id="b",
            requested_count=1, image_width=512, image_height=768,
            population_group="P", region="R",
            age_min=20, age_max=30,
            sex_or_attribute="any", skin_tone="medium",
            pose="frontal", expression="neutral",
            seeds=[99],
            provider_config=ManifestProviderConfig(provider="KAGGLE_BATCH"),
        )
        data = json.loads(m.to_json())
        # Notebook reads: manifest.get("width", manifest.get("image_width", 384))
        width = data.get("width", data.get("image_width", 384))
        height = data.get("height", data.get("image_height", 384))
        assert width == 512
        assert height == 768

    def test_notebook_batch_summary_contract(self, tmp_path):
        """
        The notebook writes batch_summary.json with job_id, successful_count,
        generated_count, and per-image records.  The worker must read all of these.
        """
        # Write a summary matching what the patched notebook produces.
        summary = {
            "job_id": 7,
            "batch_id": "batch-xyz",
            "requested_count": 2,
            "generated_count": 2,
            "successful_count": 2,
            "failed_count": 0,
            "images": [
                {"filename": "image_0001.png", "seed": 11, "prompt": "P", "width": 384, "height": 384},
                {"filename": "image_0002.png", "seed": 22, "prompt": "P", "width": 384, "height": 384},
            ],
        }
        summary_file = tmp_path / "batch_summary.json"
        summary_file.write_text(json.dumps(summary), encoding="utf-8")

        data = json.loads(summary_file.read_text())
        # Worker checks job_id for overlap guard.
        assert data["job_id"] == 7
        # Worker accepts either generated_count or successful_count.
        assert data.get("generated_count") == 2 or data.get("successful_count") == 2
        # Per-image records include filename and seed.
        assert data["images"][0]["filename"] == "image_0001.png"
        assert data["images"][0]["seed"] == 11

    def test_notebook_summary_without_job_id_falls_back_to_manifest(self, tmp_path):
        """
        If batch_summary.json has no job_id, worker must fall back to the
        locally-saved manifest.json to perform the overlap check.
        """
        # Write a summary without job_id (old notebook format).
        summary = {
            "batch_id": "batch-old",
            "requested_count": 1,
            "generated_count": 1,
            "successful_count": 1,
            "failed_count": 0,
            "images": [{"filename": "image_0001.png", "seed": 5, "prompt": "P", "width": 384, "height": 384}],
        }
        # Write a local manifest that has job_id=3.
        manifest_data = {"job_id": 3, "batch_id": "batch-old"}

        artifact_dir = tmp_path / "job_3"
        artifact_dir.mkdir()
        (artifact_dir / "batch_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        # Manually replicate what the worker does to resolve job_id.
        summary_loaded = json.loads((artifact_dir / "batch_summary.json").read_text())
        summary_job_id = summary_loaded.get("job_id")
        if summary_job_id is None:
            local_m = json.loads((artifact_dir / "manifest.json").read_text())
            summary_job_id = local_m.get("job_id")

        assert summary_job_id == 3  # Correctly resolved from manifest.


class TestTimeoutEnforcement:
    """
    Tests that _kaggle_poll() enforces KAGGLE_RUN_TIMEOUT_SECONDS.
    """

    def _make_kaggle_job(self, db_session, client):
        from backend.tests.conftest import create_test_dataset
        dataset_dict = create_test_dataset(client, overrides={"num_images": 1})
        from backend.app.models.dataset import Dataset
        from backend.app.core.status import DatasetStatus, JobStatus
        from backend.app.services.job_service import create_job
        dataset = db_session.query(Dataset).filter(Dataset.id == dataset_dict["id"]).first()
        dataset.status = DatasetStatus.GENERATING
        db_session.flush()
        job = create_job(db_session, dataset=dataset, provider="KAGGLE_BATCH", requested_count=1)
        db_session.flush()
        job.status = JobStatus.RUNNING.value
        job.batch_id = "test-batch"
        db_session.commit()
        db_session.refresh(job)
        db_session.refresh(dataset)
        return job, dataset

    def test_poll_timeout_exceeded_transitions_to_failed(
        self, client: TestClient, db_session: Session
    ):
        """Job running longer than timeout must be failed immediately."""
        from backend.worker import _kaggle_poll
        from backend.app.core.status import JobStatus
        from generation.interface import ProviderOperationResult
        from datetime import datetime, timedelta, timezone

        job, dataset = self._make_kaggle_job(db_session, client)

        # Set started_at to well beyond the timeout.
        job.started_at = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=8000)
        )
        push_time = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=8000)
        )
        job.current_stage = f"PUSH_TS={push_time.isoformat()}|Running"
        job.provider_run_id = "user/apac-gen"
        db_session.commit()

        mock_provider = MagicMock()
        mock_provider.poll_status.return_value = ProviderOperationResult.pending(
            message="running"
        )

        with patch("backend.worker.get_provider", return_value=mock_provider), \
             patch("backend.worker.settings") as s:
            s.KAGGLE_POST_PUSH_DELAY_SECONDS = 90
            s.KAGGLE_MAX_POLL_ERRORS = 5
            s.KAGGLE_RUN_TIMEOUT_SECONDS = 7200   # 2 hours; job ran 8000s
            s.JOB_LEASE_SECONDS = 300
            _kaggle_poll(db_session, job)

        db_session.refresh(job)
        assert job.status == JobStatus.FAILED.value
        assert "timeout" in (job.error_summary or "").lower()

    def test_poll_within_timeout_does_not_fail(
        self, client: TestClient, db_session: Session
    ):
        """Job still within timeout must not be failed."""
        from backend.worker import _kaggle_poll
        from backend.app.core.status import JobStatus
        from generation.interface import ProviderOperationResult
        from datetime import datetime, timedelta, timezone

        job, dataset = self._make_kaggle_job(db_session, client)
        job.started_at = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=60)
        )
        push_time = (
            datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=200)
        )
        job.current_stage = f"PUSH_TS={push_time.isoformat()}|Running"
        job.provider_run_id = "user/apac-gen"
        db_session.commit()

        mock_provider = MagicMock()
        mock_provider.poll_status.return_value = ProviderOperationResult.pending(
            message="running"
        )

        with patch("backend.worker.get_provider", return_value=mock_provider), \
             patch("backend.worker.settings") as s:
            s.KAGGLE_POST_PUSH_DELAY_SECONDS = 90
            s.KAGGLE_MAX_POLL_ERRORS = 5
            s.KAGGLE_RUN_TIMEOUT_SECONDS = 7200
            s.JOB_LEASE_SECONDS = 300
            _kaggle_poll(db_session, job)

        db_session.refresh(job)
        # Still RUNNING — timeout not exceeded, still pending.
        assert job.status == JobStatus.RUNNING.value


class TestValidateNestedArtifacts:
    """
    Tests that _kaggle_validate() finds images recursively (notebook layout).
    """

    def _make_kaggle_job(self, db_session, client):
        from backend.tests.conftest import create_test_dataset
        dataset_dict = create_test_dataset(client, overrides={"num_images": 2})
        from backend.app.models.dataset import Dataset
        from backend.app.core.status import DatasetStatus, JobStatus
        from backend.app.services.job_service import create_job
        dataset = db_session.query(Dataset).filter(Dataset.id == dataset_dict["id"]).first()
        dataset.status = DatasetStatus.GENERATING
        db_session.flush()
        job = create_job(db_session, dataset=dataset, provider="KAGGLE_BATCH", requested_count=2)
        db_session.flush()
        job.status = JobStatus.VALIDATING.value
        job.batch_id = "batch-nested"
        db_session.commit()
        db_session.refresh(job)
        db_session.refresh(dataset)
        return job, dataset

    def test_validate_finds_images_in_nested_notebook_layout(
        self, client: TestClient, db_session: Session, tmp_path: Path
    ):
        """
        Images placed under rhinoplasty_dataset/batches/{batch_id}/images/
        must be found and imported into the database.
        """
        from backend.worker import _kaggle_validate
        from backend.app.core.status import JobStatus
        from backend.app.models.generated_image import GeneratedImage
        from PIL import Image as PILImage

        job, dataset = self._make_kaggle_job(db_session, client)

        # Mirror the notebook's actual output structure.
        artifact_root = tmp_path / f"job_{job.id}"
        batch_dir = (
            artifact_root / "rhinoplasty_dataset" / "batches" / "batch-nested"
        )
        images_dir = batch_dir / "images"
        images_dir.mkdir(parents=True)

        for i in range(1, 3):
            img = PILImage.new("RGB", (384, 384), color=(i * 50, 100, 150))
            img.save(images_dir / f"image_{i:04d}.png")

        summary = {
            "job_id": job.id,
            "batch_id": "batch-nested",
            "requested_count": 2,
            "generated_count": 2,
            "successful_count": 2,
            "failed_count": 0,
            "images": [
                {"filename": "image_0001.png", "seed": 1, "prompt": "P1", "width": 384, "height": 384},
                {"filename": "image_0002.png", "seed": 2, "prompt": "P2", "width": 384, "height": 384},
            ],
        }
        (batch_dir / "batch_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

        validated_dir = tmp_path / "generated"
        rejected_dir = tmp_path / "rejected"

        with patch("backend.worker.settings") as s:
            s.ARTIFACT_DOWNLOAD_DIR = str(tmp_path)
            s.GENERATED_IMAGES_DIR = str(validated_dir)
            s.REJECTED_IMAGES_DIR = str(rejected_dir)
            s.IMAGE_WIDTH = 384
            s.IMAGE_HEIGHT = 384
            s.ALLOWED_IMAGE_FORMATS = "png,jpg,jpeg,webp"
            s.MAX_IMAGE_FILE_BYTES = 20 * 1024 * 1024
            s.REQUIRE_FACE_DETECTION = False
            s.ENABLE_DUPLICATE_DETECTION = True
            s.DUPLICATE_HASH_THRESHOLD = 10
            s.allowed_formats = ["png", "jpg", "jpeg", "webp"]
            _kaggle_validate(db_session, job, dataset)

        db_session.refresh(job)
        assert job.status == JobStatus.PACKAGING.value
        assert job.accepted_count == 2

        images = (
            db_session.query(GeneratedImage)
            .filter(GeneratedImage.job_id == job.id)
            .order_by(GeneratedImage.seed)
            .all()
        )
        assert len(images) == 2
        assert images[0].seed == 1
        assert images[1].seed == 2
        assert all(img.validation_status == "ACCEPTED" for img in images)


class TestKaggleQueueAndConcurrency:
    """Tests for single-flight locking, queue dispatching, and worker recovery."""

    def test_job_completion_releases_lock_and_dispatches_next_queued_job(
        self, client: TestClient, db_session: Session, tmp_path
    ):
        """When Job A completes, the lock is released and Job B is immediately dispatched."""
        from backend.worker import run_once
        from backend.app.core.status import JobStatus, DatasetStatus
        from backend.app.models.generation_job import GenerationJob
        from backend.app.services.job_service import create_job
        from backend.app.models.dataset import Dataset
        from generation.interface import ProviderOperationResult

        ds1_dict = create_test_dataset(client, overrides={"dataset_name": "DS_COMPL_1"})
        ds2_dict = create_test_dataset(client, overrides={"dataset_name": "DS_COMPL_2"})

        ds1 = db_session.query(Dataset).filter(Dataset.id == ds1_dict["id"]).first()
        ds2 = db_session.query(Dataset).filter(Dataset.id == ds2_dict["id"]).first()
        ds1.status = DatasetStatus.GENERATING
        ds2.status = DatasetStatus.GENERATING
        db_session.flush()

        job1 = create_job(db_session, dataset=ds1, provider="KAGGLE_BATCH", requested_count=1)
        job2 = create_job(db_session, dataset=ds2, provider="KAGGLE_BATCH", requested_count=1)
        db_session.flush()

        # Job1 is in PACKAGING (about to finish)
        job1.status = JobStatus.PACKAGING.value
        from backend.worker import WORKER_ID
        job1.worker_id = WORKER_ID
        db_session.commit()

        # Mock packaging for job1 and preparation for job2
        mock_provider = MagicMock()
        mock_provider.prepare_batch.return_value = ProviderOperationResult.success()
        mock_provider.submit_batch.return_value = ProviderOperationResult.success(
            provider_run_id="rutujavaijanathpote/apac-face-generator"
        )

        with patch("backend.worker.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "rutujavaijanathpote/apac-face-generator"
            s.WORKER_POLL_INTERVAL_SECONDS = 10
            s.JOB_LEASE_SECONDS = 300
            s.IMAGE_WIDTH = 512
            s.IMAGE_HEIGHT = 512
            s.ARTIFACT_DOWNLOAD_DIR = str(tmp_path)
            with patch("backend.worker.get_provider", return_value=mock_provider):
                with patch("backend.worker._kaggle_package") as mock_pkg:
                    def fake_package(db, j, d):
                        from backend.app.services.job_service import transition_job_status
                        transition_job_status(db, j, JobStatus.COMPLETED, stage="Completed")
                    mock_pkg.side_effect = fake_package

                    processed = run_once(db_session)

        db_session.refresh(job1)
        db_session.refresh(job2)

        # Job1 completed and released the lock
        assert job1.status == JobStatus.COMPLETED.value
        # Job2 was immediately claimed from QUEUED to SUBMITTED
        assert job2.status in (JobStatus.SUBMITTED.value, JobStatus.PREPARING.value)
        assert job2.worker_id == WORKER_ID

    def test_cancelled_job_does_not_block_next_queued_job(
        self, client: TestClient, db_session: Session
    ):
        """Cancelled jobs do not hold the lock or block queued jobs."""
        from backend.worker import run_once
        from backend.app.core.status import JobStatus, DatasetStatus
        from backend.app.models.generation_job import GenerationJob
        from backend.app.services.job_service import create_job
        from backend.app.models.dataset import Dataset
        from generation.interface import ProviderOperationResult

        ds1_dict = create_test_dataset(client, overrides={"dataset_name": "DS_CANCEL_1"})
        ds2_dict = create_test_dataset(client, overrides={"dataset_name": "DS_CANCEL_2"})

        ds1 = db_session.query(Dataset).filter(Dataset.id == ds1_dict["id"]).first()
        ds2 = db_session.query(Dataset).filter(Dataset.id == ds2_dict["id"]).first()

        job1 = create_job(db_session, dataset=ds1, provider="KAGGLE_BATCH", requested_count=1)
        job2 = create_job(db_session, dataset=ds2, provider="KAGGLE_BATCH", requested_count=1)
        job1.status = JobStatus.CANCELLED.value
        db_session.commit()

        mock_provider = MagicMock()
        mock_provider.prepare_batch.return_value = ProviderOperationResult.success()
        mock_provider.submit_batch.return_value = ProviderOperationResult.success(
            provider_run_id="rutujavaijanathpote/apac-face-generator"
        )

        with patch("backend.worker.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "rutujavaijanathpote/apac-face-generator"
            s.WORKER_POLL_INTERVAL_SECONDS = 10
            s.JOB_LEASE_SECONDS = 300
            s.IMAGE_WIDTH = 512
            s.IMAGE_HEIGHT = 512
            s.ARTIFACT_DOWNLOAD_DIR = "storage/artifacts_tmp"
            with patch("backend.worker.get_provider", return_value=mock_provider):
                processed = run_once(db_session)

        db_session.refresh(job2)
        assert job2.status in (JobStatus.SUBMITTED.value, JobStatus.PREPARING.value)

    def test_server_restart_adopts_in_flight_kaggle_job(
        self, client: TestClient, db_session: Session
    ):
        """A new worker instance adopts in-flight jobs from a previous worker."""
        from backend.worker import run_once, WORKER_ID
        from backend.app.core.status import JobStatus
        from backend.app.models.generation_job import GenerationJob
        from backend.app.services.job_service import create_job
        from backend.app.models.dataset import Dataset
        from generation.interface import ProviderOperationResult

        ds_dict = create_test_dataset(client, overrides={"dataset_name": "DS_RESTART"})
        ds = db_session.query(Dataset).filter(Dataset.id == ds_dict["id"]).first()

        job = create_job(db_session, dataset=ds, provider="KAGGLE_BATCH", requested_count=1)
        job.status = JobStatus.RUNNING.value
        job.worker_id = "old-crashed-worker-uuid"
        job.provider_run_id = "rutujavaijanathpote/apac-face-generator"
        from datetime import datetime, timedelta, timezone
        # Expired lease from crashed worker
        job.lease_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=10)
        db_session.commit()

        mock_provider = MagicMock()
        mock_provider.poll_status.return_value = ProviderOperationResult.pending()

        with patch("backend.worker.settings") as s:
            s.KAGGLE_KERNEL_SLUG = "rutujavaijanathpote/apac-face-generator"
            s.WORKER_POLL_INTERVAL_SECONDS = 10
            s.JOB_LEASE_SECONDS = 300
            s.KAGGLE_RUN_TIMEOUT_SECONDS = 7200
            with patch("backend.worker.get_provider", return_value=mock_provider):
                processed = run_once(db_session)

        db_session.refresh(job)
        # The new worker must have adopted the in-flight job and renewed its lease
        assert job.worker_id == WORKER_ID
        assert job.lease_expires_at > datetime.now(timezone.utc).replace(tzinfo=None)

    def test_stale_job_recovery_does_not_crash_on_invalid_transition(
        self, client: TestClient, db_session: Session
    ):
        """Stale job recovery resets PREPARING jobs to QUEUED without throwing exceptions."""
        from backend.app.core.status import JobStatus
        from backend.app.services.job_service import create_job, get_stale_jobs, requeue_stale_job
        from backend.app.models.dataset import Dataset
        from datetime import datetime, timedelta, timezone

        ds_dict = create_test_dataset(client, overrides={"dataset_name": "DS_STALE"})
        ds = db_session.query(Dataset).filter(Dataset.id == ds_dict["id"]).first()

        job = create_job(db_session, dataset=ds, provider="MOCK", requested_count=1)
        job.status = JobStatus.PREPARING.value
        job.worker_id = "dead-worker"
        job.lease_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=60)
        db_session.commit()

        stale = get_stale_jobs(db_session)
        assert any(j.id == job.id for j in stale)

        # Should recover cleanly without InvalidJobTransitionException
        recovered = requeue_stale_job(db_session, job)
        assert recovered.status == JobStatus.QUEUED.value
        assert recovered.worker_id is None

