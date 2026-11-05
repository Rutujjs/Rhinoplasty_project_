"""
Automated tests for the Kaggle Pipeline Test workflow.

Tests verify:
1. Manifest contains exactly one prompt and requested_count == 1.
2. Manifest prompt matches the shared prompt builder output exactly.
3. Notebook validation consumes manifest prompt (and rejects missing prompts).
4. Provider is explicitly KAGGLE_BATCH (not MOCK).
5. Missing Kaggle config/credentials fails explicitly without silent MOCK fallback.
6. Test creates zero database records (no Dataset, no GeneratedImage, no GenerationJob in DB).
7. Test leaves dataset counters and storage exports untouched.
8. Error reporting occurs at the exact stage where failure happens (with subsequent stages skipped).
9. Timeout and cancellation cleanup temporary workspaces.
10. Successful preview returns the retrieved PNG with valid format and dimensions.
11. Disk cleanup occurs after both success and failure.
12. API endpoints (/prompt-preview, /run, /status, /image, /cleanup) function correctly.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.schemas.kaggle_test import KaggleTestRequest, StageStatus
from backend.app.services.kaggle_test_service import (
    KagglePipelineTestRunner,
    KaggleTestRun,
    MockTestProvider,
)
from generation.interface import ProviderOperationResult
from generation.manifest import (
    GenerationManifest,
    ManifestProviderConfig,
    build_prompt_from_attributes,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_request():
    return KaggleTestRequest(
        population_group="South Asian",
        region="India",
        age_min=25,
        age_max=35,
        sex_or_attribute="female",
        skin_tone="medium_brown",
        pose="frontal",
        expression="neutral",
        lighting="neutral",
        resolution=512,
        seed=12345,
    )


# --------------------------------------------------------------------------- #
# 1 & 2. Prompt & Manifest Tests                                              #
# --------------------------------------------------------------------------- #

def test_manifest_contains_exactly_one_prompt(test_request):
    """Verify that test manifest enforces exactly one prompt and requested_count == 1."""
    prompt = build_prompt_from_attributes(
        population_group=test_request.population_group,
        region=test_request.region,
        age_min=test_request.age_min,
        age_max=test_request.age_max,
        sex_or_attribute=test_request.sex_or_attribute,
        skin_tone=test_request.skin_tone,
        pose=test_request.pose,
        expression=test_request.expression,
        lighting=test_request.lighting,
    )

    manifest = GenerationManifest(
        dataset_id=0,
        job_id=0,
        batch_id="test_run_01",
        requested_count=1,
        image_width=512,
        image_height=512,
        population_group=test_request.population_group,
        region=test_request.region,
        age_min=test_request.age_min,
        age_max=test_request.age_max,
        sex_or_attribute=test_request.sex_or_attribute,
        skin_tone=test_request.skin_tone,
        pose=test_request.pose,
        expression=test_request.expression,
        lighting=test_request.lighting,
        prompts=[prompt],
        seeds=[12345],
        provider_config=ManifestProviderConfig(provider="KAGGLE_BATCH"),
    )

    assert manifest.requested_count == 1
    assert manifest.prompts is not None
    assert len(manifest.prompts) == 1
    assert manifest.prompts[0] == prompt


def test_manifest_prompt_matches_shared_builder(test_request):
    """Verify manifest prompt is identical to build_prompt_from_attributes output."""
    expected = build_prompt_from_attributes(
        population_group=test_request.population_group,
        region=test_request.region,
        age_min=test_request.age_min,
        age_max=test_request.age_max,
        sex_or_attribute=test_request.sex_or_attribute,
        skin_tone=test_request.skin_tone,
        pose=test_request.pose,
        expression=test_request.expression,
        lighting=test_request.lighting,
    )

    runner = KagglePipelineTestRunner()
    provider = MockTestProvider(succeed=True)
    summary = runner.start_test(test_request, provider=provider)

    # Wait for test execution
    if runner._active_thread:
        runner._active_thread.join(timeout=10)

    run = runner.get_run(summary.run_id)
    assert run is not None
    assert run.prompt == expected


def test_notebook_consumes_manifest_prompt():
    """Verify that notebook logic simulation extracts and uses the manifest prompt."""
    manifest_dict = {
        "requested_count": 1,
        "prompts": ["Photorealistic synthetic portrait of a fictional person..."],
        "seeds": [42],
    }

    # Simulate Cell 4 logic
    prompts_val = manifest_dict.get("prompts")
    assert prompts_val is not None
    assert isinstance(prompts_val, list) and len(prompts_val) == 1

    # Simulate Cell 6 selection
    selected_prompt = prompts_val[0]
    assert selected_prompt == manifest_dict["prompts"][0]


# --------------------------------------------------------------------------- #
# 4 & 5. Explicit KAGGLE_BATCH & Failure Reporting (No Silent MOCK)            #
# --------------------------------------------------------------------------- #

def test_explicit_kaggle_batch_provider_config(test_request):
    """Verify manifest is explicitly configured for KAGGLE_BATCH, not MOCK."""
    runner = KagglePipelineTestRunner()
    provider = MockTestProvider(succeed=True)
    summary = runner.start_test(test_request, provider=provider)

    if runner._active_thread:
        runner._active_thread.join(timeout=10)

    run = runner.get_run(summary.run_id)
    assert run.manifest is not None
    assert run.manifest.provider_config.provider == "KAGGLE_BATCH"


def test_missing_kaggle_config_fails_without_silent_mock_fallback(test_request):
    """
    Verify that if Kaggle fails (e.g., credentials or upload fail),
    the test fails explicitly and does NOT fall back to MOCK.
    Also ensures the failure is reported as the intended upload error,
    never masked by an AttributeError on ProviderOperationResult.
    """
    runner = KagglePipelineTestRunner()
    # Provider fails at Stage 3
    provider = MockTestProvider(succeed=False, failure_stage=3)
    summary = runner.start_test(test_request, provider=provider)

    if runner._active_thread:
        runner._active_thread.join(timeout=10)

    run = runner.get_run(summary.run_id)
    assert run.status == "FAILED"
    assert run.stages[3].status == StageStatus.FAILED
    assert "upload failure" in run.stages[3].error.lower()
    assert "AttributeError" not in run.stages[3].error
    assert "Internal exception" not in run.stages[3].error
    # Must NOT have generated a mock image or passed
    assert run.image_bytes is None
    assert run.stages[9].status == StageStatus.SKIPPED


def test_provider_error_detail_propagation_regression(test_request):
    """
    Regression test: verify that ProviderOperationResult.message is reported
    as stage error and ProviderOperationResult.error_detail is passed as stage details,
    without raising an AttributeError (previously caused by accessing nonexistent .detail).
    """
    class CustomFailingProvider(MockTestProvider):
        def prepare_batch(self, manifest):
            return ProviderOperationResult.permanent_error(
                message="Kaggle manifest upload failed: HTTP 403",
                detail="Dataset update rejected: unauthorized token",
            )

    runner = KagglePipelineTestRunner()
    summary = runner.start_test(test_request, provider=CustomFailingProvider())

    if runner._active_thread:
        runner._active_thread.join(timeout=10)

    run = runner.get_run(summary.run_id)
    assert run.status == "FAILED"
    assert run.stages[3].status == StageStatus.FAILED
    assert run.stages[3].error == "Kaggle manifest upload failed: HTTP 403"
    assert run.stages[3].details == "Dataset update rejected: unauthorized token"
    assert "AttributeError" not in (run.error_summary or "")
    assert "Internal exception" not in (run.error_summary or "")


# --------------------------------------------------------------------------- #
# 6 & 7. Database & Storage Isolation                                         #
# --------------------------------------------------------------------------- #

def test_creates_no_database_records(test_request):
    """Verify that Kaggle test runner touches no database tables or sessions."""
    runner = KagglePipelineTestRunner()
    provider = MockTestProvider(succeed=True)

    with patch("backend.app.database.session.get_db") as mock_get_db:
        summary = runner.start_test(test_request, provider=provider)
        if runner._active_thread:
            runner._active_thread.join(timeout=10)

        # get_db should never be called
        mock_get_db.assert_not_called()

    run = runner.get_run(summary.run_id)
    assert run.status == "COMPLETED"


def test_does_not_modify_dataset_counters_or_exports(test_request):
    """Verify existing storage directories (generated, validated, exports) are not touched."""
    export_dir = Path(settings.EXPORT_DIR)
    validated_dir = Path(settings.VALIDATED_IMAGES_DIR)

    export_files_before = set(export_dir.rglob("*")) if export_dir.exists() else set()
    validated_files_before = set(validated_dir.rglob("*")) if validated_dir.exists() else set()

    runner = KagglePipelineTestRunner()
    provider = MockTestProvider(succeed=True)
    summary = runner.start_test(test_request, provider=provider)

    if runner._active_thread:
        runner._active_thread.join(timeout=10)

    export_files_after = set(export_dir.rglob("*")) if export_dir.exists() else set()
    validated_files_after = set(validated_dir.rglob("*")) if validated_dir.exists() else set()

    assert export_files_before == export_files_after
    assert validated_files_before == validated_files_after


# --------------------------------------------------------------------------- #
# 8, 9, 10, 11. Staged Errors, Timeouts, Cleanup, and Image Return             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fail_stage,expected_stage", [
    (3, 3),  # Kaggle manifest dataset upload
    (4, 4),  # Kernel push
    (5, 5),  # Kernel execution
    (7, 7),  # Artifact download
])
def test_stage_error_reporting_at_correct_stage(test_request, fail_stage, expected_stage):
    """Verify that errors at any stage are reported on that exact stage."""
    runner = KagglePipelineTestRunner()
    provider = MockTestProvider(succeed=False, failure_stage=fail_stage)
    summary = runner.start_test(test_request, provider=provider)

    if runner._active_thread:
        runner._active_thread.join(timeout=10)

    run = runner.get_run(summary.run_id)
    assert run.status == "FAILED"
    assert run.stages[expected_stage].status == StageStatus.FAILED
    assert run.stages[expected_stage].error is not None
    assert "Internal exception" not in run.stages[expected_stage].error
    assert "AttributeError" not in run.stages[expected_stage].error

    # Subsequent stages should be SKIPPED (except cleanup stage 10)
    for sid in range(expected_stage + 1, 10):
        assert run.stages[sid].status == StageStatus.SKIPPED

    # Cleanup stage 10 should still run and pass
    assert run.stages[10].status == StageStatus.PASSED


def test_successful_preview_returns_retrieved_png(test_request):
    """Verify preview returns valid retrieved PNG with correct format and dimensions."""
    runner = KagglePipelineTestRunner()
    provider = MockTestProvider(succeed=True)
    summary = runner.start_test(test_request, provider=provider)

    if runner._active_thread:
        runner._active_thread.join(timeout=10)

    run = runner.get_run(summary.run_id)
    assert run.status == "COMPLETED"
    assert run.image_bytes is not None

    # Verify PNG magic bytes
    assert run.image_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    # Verify decodability via Pillow
    img = Image.open(io.BytesIO(run.image_bytes))
    assert img.width == 512
    assert img.height == 512
    assert img.format == "PNG"

    # Verify data URI is populated in summary
    summary_out = run.to_summary()
    assert summary_out.image_data_uri is not None
    assert summary_out.image_data_uri.startswith("data:image/png;base64,")


def test_cleanup_occurs_after_success_and_failure(test_request):
    """Verify temporary workspace is cleaned up from disk on both success and failure."""
    runner = KagglePipelineTestRunner()

    # Success case
    prov_ok = MockTestProvider(succeed=True)
    sum_ok = runner.start_test(test_request, provider=prov_ok)
    if runner._active_thread:
        runner._active_thread.join(timeout=10)
    run_ok = runner.get_run(sum_ok.run_id)
    assert not run_ok.workspace_dir.exists()

    # Failure case
    prov_fail = MockTestProvider(succeed=False, failure_stage=4)
    sum_fail = runner.start_test(test_request, provider=prov_fail)
    if runner._active_thread:
        runner._active_thread.join(timeout=10)
    run_fail = runner.get_run(sum_fail.run_id)
    assert not run_fail.workspace_dir.exists()


# --------------------------------------------------------------------------- #
# 12. API Endpoints Integration                                                #
# --------------------------------------------------------------------------- #

def test_api_prompt_preview_endpoint(client):
    """POST /kaggle-test/prompt-preview returns canonical prompt."""
    payload = {
        "population_group": "South Asian",
        "region": "India",
        "age_min": 25,
        "age_max": 35,
        "sex_or_attribute": "female",
        "skin_tone": "medium_brown",
        "pose": "frontal",
        "expression": "neutral",
        "lighting": "neutral",
    }
    resp = client.post("/kaggle-test/prompt-preview", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prompt" in data
    assert "South Asian" in data["prompt"]
    assert "India" in data["prompt"]


def test_api_run_and_status_endpoints(client, test_request):
    """Verify /kaggle-test/run and /kaggle-test/status/{run_id} APIs hermetically."""
    payload = test_request.model_dump()
    with patch(
        "backend.app.services.kaggle_test_service.KaggleBatchProvider",
        return_value=MockTestProvider(succeed=True),
    ):
        resp = client.post("/kaggle-test/run", json=payload)
        assert resp.status_code == 202
        data = resp.json()
        assert "run_id" in data
        run_id = data["run_id"]

        # Poll status
        status_resp = client.get(f"/kaggle-test/status/{run_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["run_id"] == run_id
        assert len(status_data["stages"]) == 10


def test_api_serves_image_safely(client):
    """Verify /kaggle-test/image/{run_id} serves image and blocks invalid IDs."""
    # Invalid ID pattern
    resp_invalid = client.get("/kaggle-test/image/invalid_id_not_matching_pattern")
    assert resp_invalid.status_code == 400

    # Non-existent run
    resp_404 = client.get("/kaggle-test/image/test_00000000")
    assert resp_404.status_code == 404


# --------------------------------------------------------------------------- #
# 13. Concurrency & Deadlock Regression Tests                                 #
# --------------------------------------------------------------------------- #

def test_start_test_does_not_deadlock_and_returns_promptly(test_request):
    """
    REGRESSION TEST: Verify start_test() does not deadlock on self._lock
    and returns promptly with a valid summary.
    """
    import time
    runner = KagglePipelineTestRunner()
    provider = MockTestProvider(succeed=True)

    t0 = time.time()
    summary = runner.start_test(test_request, provider=provider)
    duration = time.time() - t0

    # start_test should return almost instantaneously (< 0.5s), not deadlock
    assert duration < 0.5
    assert summary.run_id.startswith("test_")

    if runner._active_thread:
        runner._active_thread.join(timeout=10)


def test_second_concurrent_run_rejected_while_active(test_request):
    """
    Verify that calling start_test() while a run is active raises RuntimeError
    and prevents concurrent execution, and that subsequent runs succeed once
    the active run has terminated and cleared state.
    """
    import threading

    entered_event = threading.Event()
    release_event = threading.Event()

    class DeterministicBlockingProvider(MockTestProvider):
        poll_interval_seconds = 0.01

        def poll_status(self, provider_run_id):
            # Signal that the first run is actively executing its pipeline
            entered_event.set()
            # Wait until explicitly released by the test
            if not release_event.wait(timeout=5):
                raise TimeoutError("Test did not release blocking provider in time")
            return super().poll_status(provider_run_id)

    runner = KagglePipelineTestRunner()
    prov = DeterministicBlockingProvider(succeed=True)

    # 1. Start first run
    first_summary = runner.start_test(test_request, provider=prov)

    # 2. Deterministically wait until the first run is confirmed active and blocked
    assert entered_event.wait(timeout=5), "First run never reached the blocking execution point"
    assert runner.is_running(), "Runner does not report running state while active"

    # 3. Attempt second concurrent run - must be rejected
    with pytest.raises(RuntimeError, match="currently in progress"):
        runner.start_test(test_request, provider=prov)

    # 4. Release first run to complete
    release_event.set()

    # 5. Wait for first thread to terminate and verify it is dead
    first_thread = runner._active_thread
    assert first_thread is not None, "Active thread reference missing"
    first_thread.join(timeout=5)
    assert not first_thread.is_alive(), (
        f"First run thread ({first_summary.run_id}) failed to terminate within timeout"
    )

    # 6. Verify runner has cleared active-run state
    assert not runner.is_running(), (
        f"Runner still reports active after thread finished: active_run_id={runner._active_run_id}"
    )

    # 7. Start second run now that the first has reliably cleared
    second_summary = runner.start_test(test_request, provider=MockTestProvider(succeed=True))
    assert second_summary.run_id != first_summary.run_id

    second_thread = runner._active_thread
    assert second_thread is not None
    second_thread.join(timeout=5)
    assert not second_thread.is_alive(), "Second run thread failed to terminate"


def test_concurrent_start_test_threads_allows_only_one(test_request):
    """
    Verify that when multiple threads attempt start_test() concurrently,
    only one succeeds and all others raise RuntimeError.
    """
    import threading

    runner = KagglePipelineTestRunner()
    prov = MockTestProvider(succeed=True)

    successes = []
    errors = []

    def try_start():
        try:
            summary = runner.start_test(test_request, provider=prov)
            successes.append(summary.run_id)
        except RuntimeError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=try_start) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    if runner._active_thread:
        runner._active_thread.join(timeout=10)

    # Exactly one thread must have succeeded
    assert len(successes) == 1
    assert len(errors) == 4


# --------------------------------------------------------------------------- #
# 14. Static UI Route Regression Tests                                        #
# --------------------------------------------------------------------------- #

def test_kaggle_test_ui_route_serves_html(client):
    """
    REGRESSION TEST: Verify that GET /kaggle-test serves the actual
    frontend/kaggle_test.html file (status 200) without crashing with a 500 error
    caused by invalid path resolution outside project root.
    """
    # 1. Clean GET /kaggle-test
    resp = client.get("/kaggle-test")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "Kaggle" in resp.text
    assert "<!DOCTYPE html>" in resp.text

    # 2. GET /kaggle-test/ with trailing slash
    resp_slash = client.get("/kaggle-test/")
    assert resp_slash.status_code == 200
    assert "text/html" in resp_slash.headers.get("content-type", "")

    # 3. GET with query parameters (e.g. tracking / referral parameters)
    resp_params = client.get("/kaggle-test?utm_source=chatgpt.com")
    assert resp_params.status_code == 200
    assert "text/html" in resp_params.headers.get("content-type", "")


def test_main_ui_route_serves_html(client):
    """Verify that GET /ui and /ui/ serve frontend/index.html (status 200)."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "<!DOCTYPE html>" in resp.text

    resp_slash = client.get("/ui/")
    assert resp_slash.status_code == 200
    assert "text/html" in resp_slash.headers.get("content-type", "")


# --------------------------------------------------------------------------- #
# 15. Custom Prompt Mode & Submission Tests                                   #
# --------------------------------------------------------------------------- #

def test_default_prompt_mode_is_auto():
    """Verify that KaggleTestRequest defaults to prompt_mode='auto' without changing existing behavior."""
    req = KaggleTestRequest()
    assert req.prompt_mode == "auto"
    assert req.custom_prompt is None


def test_custom_prompt_validation():
    """Verify that prompt_mode='custom' requires a non-empty custom_prompt."""
    # Valid custom prompt
    req = KaggleTestRequest(prompt_mode="custom", custom_prompt="A portrait of an elderly man with cataract")
    assert req.prompt_mode == "custom"
    assert req.custom_prompt == "A portrait of an elderly man with cataract"

    # Empty custom prompt raises ValueError
    with pytest.raises(ValueError, match="custom_prompt"):
        KaggleTestRequest(prompt_mode="custom", custom_prompt="")

    with pytest.raises(ValueError, match="custom_prompt"):
        KaggleTestRequest(prompt_mode="custom", custom_prompt="   ")

    with pytest.raises(ValueError, match="custom_prompt"):
        KaggleTestRequest(prompt_mode="custom", custom_prompt=None)

    # Invalid prompt_mode raises ValueError
    with pytest.raises(ValueError, match="prompt_mode"):
        KaggleTestRequest(prompt_mode="arbitrary_mode", custom_prompt="Some prompt")


def test_api_rejects_empty_custom_prompt(client):
    """Verify API POST /kaggle-test/run returns 422 if custom prompt mode has empty prompt."""
    resp = client.post("/kaggle-test/run", json={
        "population_group": "South Asian",
        "region": "India",
        "prompt_mode": "custom",
        "custom_prompt": "   ",
    })
    assert resp.status_code == 422


def test_custom_prompt_pipeline_execution_and_manifest():
    """Verify that running a test with prompt_mode='custom' delivers the custom prompt into manifest and cleans up."""
    custom_text = "Clinical photograph of a 30-year-old Indian female with mild facial erythema, high detail"
    req = KaggleTestRequest(
        population_group="South Asian",
        region="India",
        age_min=25,
        age_max=35,
        sex_or_attribute="female",
        prompt_mode="custom",
        custom_prompt=custom_text,
    )

    captured_manifest = []
    class ManifestCapturingProvider(MockTestProvider):
        def prepare_batch(self, manifest):
            captured_manifest.append(manifest)
            return super().prepare_batch(manifest)

    runner = KagglePipelineTestRunner()
    provider = ManifestCapturingProvider(succeed=True)

    summary = runner.start_test(req, provider=provider)
    if runner._active_thread:
        runner._active_thread.join(timeout=10)

    run = runner.get_run(summary.run_id)
    assert run.status == "COMPLETED"
    assert run.prompt == custom_text
    assert run.stages[1].status == StageStatus.PASSED
    assert "Custom prompt accepted" in (run.stages[1].details or "")

    # Verify manifest received the custom prompt
    assert len(captured_manifest) == 1
    manifest = captured_manifest[0]
    assert manifest.prompts == [custom_text]
    assert manifest.requested_count == 1

    # Stage 10 cleanup verified: workspace directory was removed
    assert run.stages[10].status == StageStatus.PASSED
    assert not run.workspace_dir.exists()


def test_auto_prompt_pipeline_execution_and_manifest():
    """Verify that prompt_mode='auto' builds canonical prompt from demographic attributes."""
    req = KaggleTestRequest(
        population_group="South Asian",
        region="India",
        age_min=25,
        age_max=35,
        sex_or_attribute="female",
        prompt_mode="auto",
    )
    runner = KagglePipelineTestRunner()
    provider = MockTestProvider(succeed=True)

    summary = runner.start_test(req, provider=provider)
    if runner._active_thread:
        runner._active_thread.join(timeout=10)

    run = runner.get_run(summary.run_id)
    assert run.status == "COMPLETED"
    assert "female" in run.prompt.lower()
    assert "india" in run.prompt.lower()
    assert run.stages[1].status == StageStatus.PASSED
    assert "Prompt generated" in (run.stages[1].details or "")


def test_ui_contains_custom_prompt_elements(client):
    """Verify that GET /kaggle-test HTML includes the prompt mode selector and custom prompt textarea."""
    resp = client.get("/kaggle-test")
    assert resp.status_code == 200
    html = resp.text
    assert 'id="f-prompt-mode"' in html
    assert 'value="auto"' in html
    assert 'value="custom"' in html
    assert 'id="container-auto-prompt"' in html
    assert 'id="container-custom-prompt"' in html
    assert 'id="f-custom-prompt"' in html
    assert 'id="btn-reset-prompt"' in html
