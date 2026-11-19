"""
FastAPI Routes for Languages, Countries, and Market Expansion Strategy.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Response
from fastapi.responses import FileResponse, Response

from backend.app.services.language_service import (
    STANDARDIZED_COUNTRIES,
    language_service,
)

router = APIRouter(
    prefix="/languages",
    tags=["Languages & Country Expansion"],
)


@router.get(
    "",
    summary="Get language expansion catalog",
    description="Returns full strategic catalog of languages, target countries, priorities, and economic statuses from Languages.pdf",
)
def get_languages() -> List[Dict[str, Any]]:
    return language_service.get_language_catalog()


@router.get(
    "/countries",
    summary="Get all supported countries and their mapped languages",
    description="Returns standardized country list with ISO-2, ISO-3 codes, region groups, and mapped languages.",
)
def get_countries() -> List[Dict[str, Any]]:
    return language_service.get_supported_countries()


@router.get(
    "/phases",
    summary="Get rollout phases",
    description="Returns rollout phases (Phase 1, Phase 2, Phase 3) and objectives.",
)
def get_phases() -> List[Dict[str, Any]]:
    return language_service.get_rollout_phases()


@router.get(
    "/report/summary",
    summary="Get Language Expansion Strategy summary",
    description="Returns structured summary matching the full Languages.pdf document.",
)
def get_strategy_summary() -> Dict[str, Any]:
    return language_service.get_strategy_summary()


@router.get(
    "/report/pdf",
    summary="Download Language Expansion Strategy PDF",
    description="Generates and downloads the verified Language Expansion Strategy PDF document matching Languages.pdf.",
)
def download_strategy_pdf() -> Response:
    pdf_bytes = language_service.generate_strategy_pdf()
    filename = "Faceify_Labs_Language_Expansion_Strategy.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-cache",
        },
    )
