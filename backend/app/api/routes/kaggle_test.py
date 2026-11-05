"""
API endpoints for the dedicated Kaggle Pipeline Test workflow.

Endpoints
---------
POST /kaggle-test/prompt-preview  – Generate live preview of the canonical prompt
POST /kaggle-test/run             – Launch the 10-stage test pipeline (async, DB-free)
GET  /kaggle-test/status/{run_id} – Poll stage-by-stage status, timing, and errors
GET  /kaggle-test/image/{run_id}  – Retrieve generated PNG preview safely from memory
POST /kaggle-test/cleanup/{run_id}– Manually purge test state from memory

Safety & Database Independence:
- Zero SQL operations or database dependency.
- Enforces single-flight execution (HTTP 409 if another run is active).
- Never falls back to MOCK when testing Kaggle.
- Prevents path traversal and arbitrary file access.
"""

from __future__ import annotations

import re
from pathlib import Path
from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import FileResponse, JSONResponse

from backend.app.core.config import settings

from backend.app.schemas.kaggle_test import (
    KaggleTestRequest,
    KaggleTestRunSummary,
    PromptPreviewRequest,
    PromptPreviewResponse,
)
from backend.app.services.kaggle_test_service import kaggle_test_runner
from generation.manifest import build_prompt_from_attributes

router = APIRouter(
    prefix="/kaggle-test",
    tags=["Kaggle Pipeline Test"],
)

_RUN_ID_REGEX = re.compile(r"^test_[a-fA-F0-9]{8,16}$")


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
def serve_kaggle_test_ui():
    """Serve the Kaggle Pipeline Test UI page."""
    html_dir = getattr(settings, "resolved_frontend_dir", None)
    if isinstance(html_dir, Path):
        target = html_dir / "kaggle_test.html"
        if target.exists():
            return FileResponse(str(target))
    # Route-relative fallback: synthetic/backend/app/api/routes -> parents[4] is synthetic project root
    route_fallback = Path(__file__).resolve().parents[4] / "frontend" / "kaggle_test.html"
    if route_fallback.exists():
        return FileResponse(str(route_fallback))
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="kaggle_test.html not found",
    )


@router.post(
    "/prompt-preview",
    response_model=PromptPreviewResponse,
    summary="Preview the canonical prompt built from configured attributes",
)
def preview_prompt(payload: PromptPreviewRequest):
    prompt = build_prompt_from_attributes(
        population_group=payload.population_group,
        region=payload.region,
        age_min=payload.age_min,
        age_max=payload.age_max,
        sex_or_attribute=payload.sex_or_attribute,
        skin_tone=payload.skin_tone,
        pose=payload.pose,
        expression=payload.expression,
        lighting=payload.lighting,
        medical_condition=payload.medical_condition,
        medical_severity=payload.medical_severity,
    )
    return PromptPreviewResponse(prompt=prompt)


@router.post(
    "/run",
    response_model=KaggleTestRunSummary,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a dedicated 10-stage Kaggle Pipeline Test run",
)
def start_kaggle_test(payload: KaggleTestRequest):
    if kaggle_test_runner.is_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "A Kaggle test run is already in progress. "
                "Wait for it to finish or reach terminal state."
            ),
        )

    try:
        summary = kaggle_test_runner.start_test(payload)
        return summary
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate test pipeline: {exc}",
        )


@router.get(
    "/status/{run_id}",
    response_model=KaggleTestRunSummary,
    summary="Get status and progress across all 10 stages for a test run",
)
def get_test_status(run_id: str):
    if not _RUN_ID_REGEX.match(run_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid run_id format")

    run = kaggle_test_runner.get_run(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Test run not found")

    return run.to_summary()


@router.get(
    "/image/{run_id}",
    summary="Retrieve the generated test image preview safely from memory",
)
def get_test_image(run_id: str):
    if not _RUN_ID_REGEX.match(run_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid run_id format")

    run = kaggle_test_runner.get_run(run_id)
    if not run or not run.image_bytes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Test image not found or not yet generated")

    return Response(
        content=run.image_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Content-Disposition": f'inline; filename="preview_{run_id}.png"',
        },
    )


@router.post(
    "/cleanup/{run_id}",
    summary="Purge in-memory image bytes and metadata for a completed run",
)
def cleanup_test_memory(run_id: str):
    if not _RUN_ID_REGEX.match(run_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid run_id format")

    run = kaggle_test_runner.get_run(run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Test run not found")

    run.image_bytes = None
    return {"message": f"Run {run_id} preview memory cleared"}
