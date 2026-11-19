"""
KaggleBatchProvider – submits a Kaggle notebook run and retrieves output.

Kaggle API behaviour (verified from official documentation):
------------------------------------------------------------
1.  PUSH: `kaggle kernels push -p <dir>` pushes a new version of the kernel
    identified by `kernel-metadata.json`'s "id" field ("username/slug").
    There is NO per-run UUID returned.  The kernel slug IS the run handle.

2.  OVERLAP GUARD: Because a single slug identifies a single execution slot,
    submitting a second job while another is running would overwrite the first
    and corrupt output attribution.  The worker enforces single-flight
    submission via the job state machine (only one KAGGLE_BATCH job may be in
    SUBMITTED/RUNNING state at a time per kernel slug).

3.  POLL: `kaggle kernels status <user>/<slug>` returns a dict whose "status"
    field is one of: "queued", "running", "complete", "error",
    "cancelRequested", "cancelled".
    Wait at least KAGGLE_POST_PUSH_DELAY_SECONDS (~90 s) after push before
    first poll — the API may briefly show the previous run's status.

4.  OUTPUT: `kaggle kernels output <user>/<slug> -p <dest>` downloads all
    files from /kaggle/working/ as they were after the most recent run.
    This downloads a ZIP which the SDK unpacks into <dest>.

5.  CANCELLATION: No CLI cancel command exists.  request_cancellation
    returns UNSUPPORTED; users must cancel via the Kaggle web UI.

Manifest delivery:
------------------
The manifest JSON is delivered to the notebook by uploading it as a new
version of a private Kaggle dataset (KAGGLE_MANIFEST_DATASET_SLUG).
The notebook reads it from:
    /kaggle/input/<manifest-dataset-slug>/manifest.json

Output contract (batch_summary.json):
--------------------------------------
The notebook must write /kaggle/working/batch_summary.json with:
{
  "job_id": <int>,
  "batch_id": "<str>",
  "requested_count": <int>,
  "successful_count": <int>,
  "failed_count": <int>,
  "images": [
    {
      "filename": "image_0001.png",
      "seed": <int>,
      "prompt": "<str>",
      "width": <int>,
      "height": <int>
    },
    ...
  ]
}

Credentials:
------------
Read from ~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY env vars.
Never stored in source code or logs.
"""

from __future__ import annotations

import builtins
import io
import json
import logging
import os
import re
import shutil
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from backend.app.core.config import settings
from backend.app.schemas.generation import GeneratedImageResult, GenerationRequest
from generation.interface import FaceGenerator, ProviderOperationResult, ProviderOutcome
from generation.manifest import GenerationManifest

logger = logging.getLogger(__name__)

# Kaggle status strings → ProviderOutcome mapping.
_KAGGLE_STATUS_MAP: dict[str, ProviderOutcome] = {
    # Pending / Running statuses
    "queued":               ProviderOutcome.PENDING,
    "running":              ProviderOutcome.PENDING,
    "new_script":           ProviderOutcome.PENDING,
    "newscript":            ProviderOutcome.PENDING,
    "cancel_requested":     ProviderOutcome.PENDING,
    "cancelrequested":      ProviderOutcome.PENDING,

    # Success statuses
    "complete":             ProviderOutcome.SUCCESS,
    "completed":            ProviderOutcome.SUCCESS,

    # Error / Failure / Cancelled statuses
    "error":                ProviderOutcome.PERMANENT_ERROR,
    "failed":               ProviderOutcome.PERMANENT_ERROR,
    "cancelled":            ProviderOutcome.PERMANENT_ERROR,
    "canceled":             ProviderOutcome.PERMANENT_ERROR,
    "cancel_acknowledged":  ProviderOutcome.PERMANENT_ERROR,
    "cancelacknowledged":   ProviderOutcome.PERMANENT_ERROR,
}


def normalize_kernel_status(status_val: Any) -> str:
    """
    Safely normalize any Kaggle kernel status value into a canonical lowercase string.

    Supports:
      - Enum instances (e.g. KernelWorkerStatus.COMPLETE -> "complete")
        Prefers the enum's .name or string .value rather than relying on str(enum).
      - Response objects with a .status attribute (e.g. ApiGetKernelSessionStatusResponse)
      - Dictionaries with a 'status' key (e.g. {"status": "complete"})
      - Plain strings with or without enum class prefixes ("KernelWorkerStatus.COMPLETE" -> "complete")
    """
    if status_val is None:
        return ""

    # If an entire response object or dict is passed, extract the status field first
    raw = status_val
    if hasattr(status_val, "status"):
        raw = status_val.status
    elif isinstance(status_val, dict) and "status" in status_val:
        raw = status_val["status"]

    # 1. Prefer .name if available (standard Enum property, e.g. KernelWorkerStatus.COMPLETE.name == "COMPLETE")
    name_attr = getattr(raw, "name", None)
    if isinstance(name_attr, str) and name_attr:
        clean = name_attr.strip().lower()
    else:
        # 2. Check if .value is a non-empty string
        val_attr = getattr(raw, "value", None)
        if isinstance(val_attr, str) and val_attr:
            clean = val_attr.strip().lower()
        else:
            clean = str(raw).strip().lower()

    # 3. If string was formatted as "kernelworkerstatus.complete" or "KernelWorkerStatus.COMPLETE",
    # strip any enum class prefix before the last dot.
    if "." in clean:
        clean = clean.split(".")[-1].strip()

    return clean



def _sanitize_error_detail(exc: Exception) -> str:
    """Extract and sanitize exception message to remove any tokens or secrets."""
    try:
        msg = str(exc).strip()
    except Exception:
        msg = repr(exc)
    if not msg:
        return ""
    # Redact potential authorization tokens / secrets / API keys (long hex or alphanumeric strings)
    msg = re.sub(r"\b[A-Fa-f0-9]{32,}\b", "[REDACTED]", msg)
    msg = re.sub(r"(key|token|secret|password)[\s:=]+[^\s]+", r"\1=[REDACTED]", msg, flags=re.IGNORECASE)
    # Ensure string contains printable characters and does not crash on console output
    msg = "".join(ch if ch.isprintable() or ch in " \t\n\r" else f"\\x{ord(ch):02x}" for ch in msg)
    return msg


@contextmanager
def _utf8_open_context():
    """
    Context manager ensuring builtins.open defaults to encoding='utf-8'
    for text mode when no encoding is explicitly specified.
    Safely protects third-party SDK calls (such as KaggleApi.kernels_push)
    from Windows ANSI/cp1252 charmap decoding errors.
    """
    orig_open = builtins.open

    def utf8_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in mode and encoding is None:
            encoding = "utf-8"
        return orig_open(file, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline, closefd=closefd, opener=opener)

    builtins.open = utf8_open
    try:
        yield
    finally:
        builtins.open = orig_open


def _get_kaggle_credentials() -> tuple[str, str]:
    """
    Return (username, api_key) for the Kaggle API.

    Resolution order (same as the official kaggle SDK):
      1. KAGGLE_USERNAME + KAGGLE_KEY environment variables.
      2. ~/.kaggle/kaggle.json  ({"username": "...", "key": "..."})

    Raises EnvironmentError with an actionable message if credentials
    are not found.  Never logs credential values.
    """
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if username and key:
        return username, key

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            data = json.loads(kaggle_json.read_text(encoding="utf-8"))
            file_username = data.get("username", "").strip()
            file_key = data.get("key", "").strip()
            if file_username and file_key:
                return file_username, file_key
        except Exception as exc:
            raise EnvironmentError(
                f"Found ~/.kaggle/kaggle.json but could not parse it: {exc}. "
                "Download a fresh token from https://www.kaggle.com/settings → API → Create New Token"
            ) from exc

    raise EnvironmentError(
        "Kaggle credentials not found. Configure one of:\n"
        "  1. Set KAGGLE_USERNAME and KAGGLE_KEY in backend/.env\n"
        "  2. Place kaggle.json at ~/.kaggle/kaggle.json\n"
        "     (download from https://www.kaggle.com/settings → API → Create New Token)"
    )


def _upload_manifest_via_rest_api(
    tmp_dir: str,
    manifest_dataset_slug: str,
    version_notes: str,
    job_id: int,
) -> None:
    """
    Upload a new version of a Kaggle dataset using the Kaggle v1 REST API
    directly via HTTP (no kaggle SDK used — works with any installed version).

    Flow:
      1. Read credentials from env / kaggle.json.
      2. Zip the contents of tmp_dir.
      3. POST the zip to /api/v1/datasets/{owner}/{slug}/versions/create
         using HTTP Basic Auth (username:key).

    Raises RuntimeError on API or HTTP failure.
    """
    import requests  # stdlib-safe; always present in this project

    username, key = _get_kaggle_credentials()
    owner_slug, dataset_slug = manifest_dataset_slug.split("/", 1)

    # Build a zip of everything in tmp_dir.
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in Path(tmp_dir).rglob("*"):
            if item.is_file():
                zf.write(item, item.relative_to(tmp_dir))
    zip_bytes = zip_buf.getvalue()

    # Step 1: upload the blob file.
    upload_url = (
        f"https://www.kaggle.com/api/v1/datasets/{owner_slug}/{dataset_slug}"
        f"/upload/file/{len(zip_bytes)}"
    )
    logger.debug(
        "prepare_batch: uploading zip (%d bytes) for job=%d to blob endpoint",
        len(zip_bytes),
        job_id,
    )
    upload_resp = requests.post(
        upload_url,
        auth=(username, key),
        data=zip_bytes,
        headers={"Content-Type": "application/zip"},
        timeout=120,
    )
    if upload_resp.status_code not in (200, 201):
        raise RuntimeError(
            f"File upload step failed: HTTP {upload_resp.status_code} — {upload_resp.text[:300]}"
        )

    blob_token = upload_resp.json().get("token", "")
    if not blob_token:
        raise RuntimeError(
            f"File upload succeeded but no token returned. Response: {upload_resp.text[:300]}"
        )

    # Step 2: create a new dataset version referencing the uploaded blob.
    version_url = (
        f"https://www.kaggle.com/api/v1/datasets/{owner_slug}/{dataset_slug}/versions"
    )
    version_payload = {
        "versionNotes": version_notes,
        "files": [{"token": blob_token}],
        "convertToCsv": False,
        "deleteOldVersions": False,
    }
    version_resp = requests.post(
        version_url,
        auth=(username, key),
        json=version_payload,
        timeout=60,
    )
    if version_resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Version creation step failed: HTTP {version_resp.status_code} — {version_resp.text[:300]}"
        )

    logger.debug(
        "prepare_batch: version created for job=%d, response status=%d",
        job_id,
        version_resp.status_code,
    )


def _get_kaggle_api():
    """
    Return an authenticated KaggleApi instance (used for kernel push/poll/output).

    Raises EnvironmentError if credentials are not found.
    Never logs credential values.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except OSError as exc:
        raise EnvironmentError(
            "Kaggle credentials not found. "
            "Ensure one of the following is configured:\n"
            "  1. Place kaggle.json at ~/.kaggle/kaggle.json\n"
            "     (download from https://www.kaggle.com/settings → API → Create New Token)\n"
            "  2. Set KAGGLE_USERNAME and KAGGLE_KEY in backend/.env\n"
            "     KAGGLE_USERNAME=<your-kaggle-username>\n"
            "     KAGGLE_KEY=<your-32-char-api-key>"
        ) from exc
    except ImportError as exc:
        raise EnvironmentError(
            "The 'kaggle' package is not installed in the active Python environment. "
            "Activate the project .venv and run: pip install 'kaggle>=1.7,<2'"
        ) from exc


def _resolve_notebook_path(settings_obj: Any) -> Path:
    resolved = getattr(settings_obj, "resolved_notebook_path", None)
    if isinstance(resolved, Path):
        return resolved
    raw = getattr(settings_obj, "KAGGLE_NOTEBOOK_PATH", None)
    if isinstance(raw, (str, Path)):
        p = Path(raw)
        if p.exists():
            return p.resolve()
        base_dir = getattr(settings_obj, "BASE_DIR", None)
        if isinstance(base_dir, Path) and (base_dir / p).exists():
            return (base_dir / p).resolve()
        return p
    return Path(str(raw or ""))


def _resolve_kernel_metadata_path(settings_obj: Any) -> Path:
    resolved = getattr(settings_obj, "resolved_kernel_metadata_path", None)
    if isinstance(resolved, Path):
        return resolved
    raw = getattr(settings_obj, "KAGGLE_KERNEL_METADATA_PATH", None)
    if isinstance(raw, (str, Path)):
        p = Path(raw)
        if p.exists():
            return p.resolve()
        base_dir = getattr(settings_obj, "BASE_DIR", None)
        if isinstance(base_dir, Path) and (base_dir / p).exists():
            return (base_dir / p).resolve()
        return p
    return Path(str(raw or ""))


class KaggleBatchProvider(FaceGenerator):
    """
    Batch image-generation provider backed by a Kaggle notebook.

    Does NOT support single-image synchronous generation.  All work goes
    through the multi-step batch API called by the background worker.
    """

    @property
    def provider_name(self) -> str:
        return "KAGGLE_BATCH"

    # ------------------------------------------------------------------ #
    # Synchronous generate – NOT supported                                 #
    # ------------------------------------------------------------------ #

    def generate(self, request: GenerationRequest) -> GeneratedImageResult:
        raise NotImplementedError(
            "KaggleBatchProvider does not support synchronous single-image "
            "generation. Use the worker batch API instead."
        )

    # ------------------------------------------------------------------ #
    # prepare_batch                                                        #
    # ------------------------------------------------------------------ #

    def prepare_batch(self, manifest: GenerationManifest) -> ProviderOperationResult:
        """
        Validate credentials and upload the manifest JSON as a new version
        of the private manifest dataset on Kaggle.

        The notebook reads the manifest from:
            /kaggle/input/<manifest-dataset-slug>/manifest.json

        Steps:
          1. Verify KAGGLE_MANIFEST_DATASET_SLUG is configured.
          2. Authenticate with the Kaggle API.
          3. Write manifest.json into a temp directory.
          4. Call dataset_create_version() to upload it.
          5. Return SUCCESS or PERMANENT_ERROR.
        """
        manifest_dataset_slug = settings.KAGGLE_MANIFEST_DATASET_SLUG
        if not manifest_dataset_slug:
            return ProviderOperationResult.permanent_error(
                message=(
                    "KAGGLE_MANIFEST_DATASET_SLUG is not configured. "
                    "Create a private Kaggle dataset and set this variable."
                ),
            )

        logger.info(
            "prepare_batch: uploading manifest for job=%d batch=%s to dataset %s",
            manifest.job_id,
            manifest.batch_id,
            manifest_dataset_slug,
        )

        try:
            api = _get_kaggle_api()
        except EnvironmentError as exc:
            return ProviderOperationResult.permanent_error(
                message=str(exc),
            )

        # Write manifest and dataset metadata to a temporary directory for upload.
        with tempfile.TemporaryDirectory(prefix="apac_manifest_") as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            manifest.save(manifest_path)

            # Kaggle API's dataset_create_version requires a dataset-metadata.json
            # containing at least the dataset "id" (owner/dataset-slug) in the upload folder.
            dataset_meta = {
                "id": manifest_dataset_slug,
            }
            (Path(tmp_dir) / "dataset-metadata.json").write_text(
                json.dumps(dataset_meta, indent=2), encoding="utf-8"
            )

            try:
                with _utf8_open_context():
                    api.dataset_create_version(
                        folder=tmp_dir,
                        version_notes=(
                            f"job_id={manifest.job_id} "
                            f"batch_id={manifest.batch_id}"
                        ),
                        quiet=True,
                        convert_to_csv=False,
                        delete_old_versions=False,
                    )
            except Exception as exc:
                safe_detail = _sanitize_error_detail(exc)

                # Provide specific, actionable messages for common failure modes.
                exc_str = str(exc)
                if "401" in exc_str or "Unauthorized" in exc_str:
                    action = (
                        "HTTP 401 Unauthorized from Kaggle blob upload API. "
                        "This usually means one of:\n"
                        "  1. The application is running under the wrong Python environment "
                        "(system kaggle 2.x instead of project .venv kaggle 1.7.x). "
                        "Start the app with: .venv\\Scripts\\uvicorn backend.app.main:app --reload\n"
                        "  2. The Kaggle API key in ~/.kaggle/kaggle.json is expired or revoked. "
                        "Regenerate it at https://www.kaggle.com/settings → API → Create New Token\n"
                        "  3. KAGGLE_USERNAME and KAGGLE_KEY are not set in backend/.env and "
                        "~/.kaggle/kaggle.json is not accessible from the server process."
                    )
                    logger.error(
                        "prepare_batch: dataset upload UNAUTHORIZED for job %d. %s",
                        manifest.job_id,
                        action,
                    )
                    return ProviderOperationResult.permanent_error(
                        message=action,
                        detail=safe_detail,
                    )
                elif "403" in exc_str or "Forbidden" in exc_str:
                    action = (
                        f"HTTP 403 Forbidden: authenticated user does not have write permission "
                        f"on dataset '{manifest_dataset_slug}'. "
                        "Ensure the dataset is owned by the authenticated Kaggle account."
                    )
                    logger.error(
                        "prepare_batch: dataset upload FORBIDDEN for job %d on slug %s",
                        manifest.job_id,
                        manifest_dataset_slug,
                    )
                    return ProviderOperationResult.permanent_error(
                        message=action,
                        detail=safe_detail,
                    )
                elif "404" in exc_str or "Not Found" in exc_str:
                    action = (
                        f"HTTP 404 Not Found: dataset '{manifest_dataset_slug}' does not exist "
                        "on Kaggle. Create it manually via the Kaggle web UI before generating."
                    )
                    logger.error(
                        "prepare_batch: dataset not found for job %d: %s",
                        manifest.job_id,
                        manifest_dataset_slug,
                    )
                    return ProviderOperationResult.permanent_error(
                        message=action,
                        detail=safe_detail,
                    )

                # Generic upload failure.
                logger.error(
                    "prepare_batch: dataset upload failed for job %d: %s%s",
                    manifest.job_id,
                    type(exc).__name__,
                    f" ({safe_detail})" if safe_detail else "",
                )
                err_msg = (
                    f"Manifest dataset upload failed: {type(exc).__name__}: {safe_detail}"
                    if safe_detail
                    else f"Manifest dataset upload failed: {type(exc).__name__}"
                )
                return ProviderOperationResult.permanent_error(
                    message=err_msg,
                    detail=safe_detail,
                )

        logger.info(
            "prepare_batch: manifest uploaded successfully for job=%d",
            manifest.job_id,
        )
        return ProviderOperationResult.success(
            message="Manifest uploaded to Kaggle dataset",
        )

    # ------------------------------------------------------------------ #
    # submit_batch                                                         #
    # ------------------------------------------------------------------ #

    def submit_batch(self, manifest: GenerationManifest) -> ProviderOperationResult:
        """
        Trigger a new kernel run by pushing the real .ipynb notebook together
        with a kernel-metadata.json that attaches the manifest dataset as an
        input.

        The notebook path and kernel-metadata path are read from config so
        the researcher can swap notebooks without touching provider code.
        A backup of the notebook is NOT made by this method — the file is
        read-only; Kaggle receives a copy, the local file is unmodified.

        The `provider_run_id` stored in the DB is the kernel slug
        ("username/kernel-slug"), which is the only stable handle for
        polling and output retrieval.

        Overlap guard: the worker state machine ensures only one job per slug
        is in SUBMITTED/RUNNING state at any time.
        """
        import shutil as _shutil

        kernel_slug = settings.KAGGLE_KERNEL_SLUG
        if not kernel_slug:
            return ProviderOperationResult.permanent_error(
                message="KAGGLE_KERNEL_SLUG is not configured.",
            )

        manifest_dataset_slug = settings.KAGGLE_MANIFEST_DATASET_SLUG
        if not manifest_dataset_slug:
            return ProviderOperationResult.permanent_error(
                message="KAGGLE_MANIFEST_DATASET_SLUG is not configured.",
            )

        # Verify the notebook exists before trying to push.
        notebook_path = _resolve_notebook_path(settings)
        if not notebook_path.exists():
            return ProviderOperationResult.permanent_error(
                message=(
                    f"Notebook file not found: {notebook_path}. "
                    "Set KAGGLE_NOTEBOOK_PATH in .env to the correct path."
                ),
            )

        # Load the reference kernel-metadata.json, patch in the manifest
        # dataset source, and write into a temp push directory alongside
        # a copy of the notebook.  The original files are never modified.
        metadata_path = _resolve_kernel_metadata_path(settings)
        if not metadata_path.exists():
            return ProviderOperationResult.permanent_error(
                message=(
                    f"kernel-metadata.json not found: {metadata_path}. "
                    "Set KAGGLE_KERNEL_METADATA_PATH in .env to the correct path."
                ),
            )

        try:
            base_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return ProviderOperationResult.permanent_error(
                message=f"Could not read kernel-metadata.json: {type(exc).__name__}",
                detail=str(exc),
            )

        logger.info(
            "submit_batch: pushing kernel %s for job=%d (notebook=%s)",
            kernel_slug,
            manifest.job_id,
            notebook_path.name,
        )

        try:
            api = _get_kaggle_api()
        except EnvironmentError as exc:
            return ProviderOperationResult.permanent_error(message=str(exc))

        with tempfile.TemporaryDirectory(prefix="apac_kernel_") as tmp_dir:
            tmp = Path(tmp_dir)

            # Copy the notebook into the push directory.
            # To prevent Windows charmap / cp1252 UnicodeDecodeError inside third-party SDK calls
            # (which invoke open(code_file) without specifying an encoding), ensure_ascii=True
            # is applied for .ipynb JSON documents. This preserves all Unicode characters via
            # standard ASCII JSON escapes (\uXXXX) without corrupting the file or losing fidelity.
            dest_nb = tmp / notebook_path.name
            try:
                if notebook_path.suffix.lower() == ".ipynb":
                    nb_content = notebook_path.read_text(encoding="utf-8")
                    nb_json = json.loads(nb_content)
                    dest_nb.write_text(
                        json.dumps(nb_json, ensure_ascii=True, indent=1),
                        encoding="utf-8",
                    )
                else:
                    _shutil.copy2(notebook_path, dest_nb)
            except Exception:
                _shutil.copy2(notebook_path, dest_nb)

            # Build patched metadata: ensure manifest dataset is attached,
            # GPU is enabled, internet is enabled, and the code_file points
            # at the notebook we just copied.
            dataset_sources = list(base_meta.get("dataset_sources", []))
            if manifest_dataset_slug not in dataset_sources:
                dataset_sources.append(manifest_dataset_slug)

            patched_meta = {
                **base_meta,
                "id": kernel_slug,
                "code_file": notebook_path.name,
                "kernel_type": "notebook",
                "language": "python",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": dataset_sources,
            }
            (tmp / "kernel-metadata.json").write_text(
                json.dumps(patched_meta, indent=2, ensure_ascii=True), encoding="utf-8"
            )

            try:
                with _utf8_open_context():
                    api.kernels_push(str(tmp))
            except Exception as exc:
                safe_detail = _sanitize_error_detail(exc)
                logger.error(
                    "submit_batch: kernel push failed for job %d: %s%s",
                    manifest.job_id,
                    type(exc).__name__,
                    f" ({safe_detail})" if safe_detail else "",
                )
                err_msg = (
                    f"Kernel push failed: {type(exc).__name__}: {safe_detail}"
                    if safe_detail
                    else f"Kernel push failed: {type(exc).__name__}"
                )
                return ProviderOperationResult.permanent_error(
                    message=err_msg,
                    detail=safe_detail,
                )

        logger.info(
            "submit_batch: kernel %s pushed for job=%d",
            kernel_slug,
            manifest.job_id,
        )
        return ProviderOperationResult.success(
            provider_run_id=kernel_slug,
            message=f"Kernel {kernel_slug} submitted",
        )

    # ------------------------------------------------------------------ #
    # poll_status                                                          #
    # ------------------------------------------------------------------ #

    def poll_status(self, provider_run_id: str) -> ProviderOperationResult:
        """
        Poll Kaggle for the status of the most recent run of this kernel.

        provider_run_id is the kernel slug ("username/kernel-slug").

        Kaggle may return the status of the PREVIOUS run for ~90 seconds
        after a push.  The worker handles this by waiting
        KAGGLE_POST_PUSH_DELAY_SECONDS before the first poll.
        """
        kernel_slug = provider_run_id
        if not kernel_slug:
            return ProviderOperationResult.permanent_error(
                message="provider_run_id (kernel slug) is empty",
            )

        logger.debug("poll_status: checking kernel %s", kernel_slug)

        try:
            api = _get_kaggle_api()
        except EnvironmentError as exc:
            return ProviderOperationResult.transient_error(
                message=f"Kaggle auth failed during poll: {type(exc).__name__}",
                detail=str(exc),
            )

        try:
            status_response = api.kernels_status(kernel_slug)
        except Exception as exc:
            logger.warning(
                "poll_status: API call failed for %s: %s",
                kernel_slug,
                type(exc).__name__,
            )
            return ProviderOperationResult.transient_error(
                message=f"Status poll failed: {type(exc).__name__}",
                detail=str(exc),
            )

        # Normalise status safely from enum, object, or dict.
        raw_status = normalize_kernel_status(status_response)
        logger.info("poll_status: kernel %s → %r", kernel_slug, raw_status)

        outcome = _KAGGLE_STATUS_MAP.get(raw_status)
        if outcome is None:
            # Check for recognized keywords in unmapped or compound statuses
            if any(term in raw_status for term in ("pending", "queued", "running")):
                outcome = ProviderOutcome.PENDING
            elif any(term in raw_status for term in ("error", "fail", "cancel")):
                outcome = ProviderOutcome.PERMANENT_ERROR

        if outcome is None:
            # Unknown status — must not be treated as success.
            logger.warning(
                "poll_status: unrecognised status %r for %s, treating as PENDING",
                raw_status,
                kernel_slug,
            )
            return ProviderOperationResult.pending(
                message=f"Unrecognised status: {raw_status!r}",
            )

        if outcome == ProviderOutcome.PENDING:
            return ProviderOperationResult.pending(
                message=f"Kernel status: {raw_status}",
            )
        if outcome == ProviderOutcome.SUCCESS:
            return ProviderOperationResult.success(
                provider_run_id=kernel_slug,
                message="Kernel completed successfully",
            )

        # Permanent error (error, failed, cancelled)
        failure_msg = ""
        if hasattr(status_response, "failure_message") and status_response.failure_message:
            failure_msg = str(status_response.failure_message).strip()
        elif hasattr(status_response, "failureMessage") and status_response.failureMessage:
            failure_msg = str(status_response.failureMessage).strip()
        elif isinstance(status_response, dict):
            failure_msg = str(status_response.get("failureMessage") or status_response.get("failure_message") or "").strip()

        err_msg = f"Kernel ended with status: {raw_status}"
        if failure_msg:
            err_msg += f" ({failure_msg})"

        return ProviderOperationResult.permanent_error(
            message=err_msg,
            detail=failure_msg or None,
        )

    # ------------------------------------------------------------------ #
    # retrieve_artifacts                                                   #
    # ------------------------------------------------------------------ #

    def retrieve_artifacts(
        self,
        provider_run_id: str,
        destination_dir: str,
        manifest: GenerationManifest | None,
    ) -> ProviderOperationResult:
        """
        Download all /kaggle/working/ output files from the completed run.

        Kaggle packages the working directory as a ZIP and the SDK unpacks
        it into destination_dir.  The notebook saves images under:
            rhinoplasty_dataset/batches/{batch_id}/images/

        and the summary at:
            rhinoplasty_dataset/batches/{batch_id}/batch_summary.json

        This method scans recursively to find all images and the summary
        regardless of nesting depth.

        Returns SUCCESS with data:
          "artifact_paths" : list[str]  – absolute paths to image files
          "summary_path"   : str | None – absolute path to batch_summary.json

        Artifact isolation: files are downloaded into a job-specific
        subdirectory (destination_dir = storage/artifacts_tmp/job_{id}/).
        Because the kernel slug is single-flight (only one job runs at a
        time), there is no risk of mixing artifacts from concurrent jobs.
        """
        kernel_slug = provider_run_id
        dest = Path(destination_dir)
        dest.mkdir(parents=True, exist_ok=True)

        logger.info(
            "retrieve_artifacts: downloading output for kernel %s → %s",
            kernel_slug,
            dest,
        )

        try:
            api = _get_kaggle_api()
        except EnvironmentError as exc:
            return ProviderOperationResult.permanent_error(message=str(exc))

        try:
            with _utf8_open_context():
                api.kernels_output(kernel_slug, path=str(dest), quiet=True)
        except Exception as exc:
            safe_detail = _sanitize_error_detail(exc)
            logger.error(
                "retrieve_artifacts: download failed for %s: %s%s",
                kernel_slug,
                type(exc).__name__,
                f" ({safe_detail})" if safe_detail else "",
            )
            err_msg = (
                f"Artifact download failed: {type(exc).__name__}: {safe_detail}"
                if safe_detail
                else f"Artifact download failed: {type(exc).__name__}"
            )
            return ProviderOperationResult.transient_error(
                message=err_msg,
                detail=safe_detail,
            )

        # Scan recursively — images and summary may be nested under
        # rhinoplasty_dataset/batches/{batch_id}/
        image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        artifact_paths: list[str] = []
        summary_path: str | None = None

        for f in sorted(dest.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() in image_extensions:
                artifact_paths.append(str(f.resolve()))
            elif f.name == "batch_summary.json" and summary_path is None:
                # Take the first (should be only one) batch_summary.json found.
                summary_path = str(f.resolve())

        if not artifact_paths and summary_path is None:
            return ProviderOperationResult.permanent_error(
                message=(
                    "No image files or batch_summary.json found in downloaded "
                    f"output for kernel {kernel_slug}. "
                    "Ensure the notebook writes all outputs under /kaggle/working/."
                ),
            )

        logger.info(
            "retrieve_artifacts: found %d image(s), summary=%s",
            len(artifact_paths),
            "yes" if summary_path else "no",
        )

        return ProviderOperationResult.success(
            provider_run_id=kernel_slug,
            data={
                "artifact_paths": artifact_paths,
                "summary_path": summary_path,
            },
            message=f"Downloaded {len(artifact_paths)} image(s)",
        )

    # ------------------------------------------------------------------ #
    # request_cancellation                                                 #
    # ------------------------------------------------------------------ #

    def request_cancellation(self, provider_run_id: str) -> ProviderOperationResult:
        """
        Kaggle has no CLI or API cancel command.

        Returns UNSUPPORTED — the user must cancel via the Kaggle web UI.
        The worker will transition the job to CANCEL_REQUESTED and the
        polling loop will detect when the kernel reaches "cancelled" status.
        """
        logger.info(
            "request_cancellation: no Kaggle API cancel endpoint; "
            "user must cancel kernel %s via the Kaggle web UI",
            provider_run_id,
        )
        return ProviderOperationResult.unsupported(
            operation="request_cancellation",
            provider=self.provider_name,
        )
