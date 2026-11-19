"""
FastAPI application entry point.

All routers are registered here.  New routers should be added at the bottom
of this file following the existing pattern.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.app.database.session import get_db

from backend.app.api.routes.datasets import router as dataset_router
from backend.app.api.routes.generation import router as generation_router
from backend.app.api.routes.images import router as image_router
from backend.app.api.routes.jobs import router as jobs_router
from backend.app.api.routes.kaggle_test import router as kaggle_test_router
from backend.app.api.routes.deformation import (
    router as deformation_router,
    global_deformation_router,
)
from backend.app.api.routes.languages import router as languages_router
from backend.app.core.config import BASE_DIR, settings
from backend.app.core.exceptions import AppException

# Resolve frontend directory reliably from project root
FRONTEND_DIR = getattr(settings, "resolved_frontend_dir", None)
if not isinstance(FRONTEND_DIR, Path) or not FRONTEND_DIR.exists():
    FRONTEND_DIR = (BASE_DIR / "frontend").resolve()
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = (Path(__file__).resolve().parents[2] / "frontend").resolve()


@asynccontextmanager
async def lifespan(app: FastAPI):
    is_testing = (
        "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
        or os.environ.get("RUN_EMBEDDED_WORKER") == "false"
        or not getattr(settings, "RUN_EMBEDDED_WORKER", False)
    )
    if not is_testing:
        from backend.worker import start_embedded_worker, stop_embedded_worker
        start_embedded_worker()
        try:
            yield
        finally:
            stop_embedded_worker()
    else:
        yield


app = FastAPI(
    lifespan=lifespan,
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Privacy-safe synthetic APAC patient face dataset generator. "
        "Supports MOCK (local) and KAGGLE_BATCH generation providers."
    ),
)


# --------------------------------------------------------------------------- #
# Exception handler                                                            #
# --------------------------------------------------------------------------- #

@app.exception_handler(AppException)
async def app_exception_handler(
    request: Request,
    exc: AppException,
):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message},
    )


# --------------------------------------------------------------------------- #
# Health endpoints                                                             #
# --------------------------------------------------------------------------- #

@app.get("/")
def root():
    return {
        "message": "Synthetic APAC Patient Face Dataset Generator API",
        "version": settings.APP_VERSION,
        "status": "running",
        "provider": settings.effective_provider,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "provider": settings.effective_provider,
        "kaggle_kernel": settings.KAGGLE_KERNEL_SLUG,
        "kaggle_manifest": settings.KAGGLE_MANIFEST_DATASET_SLUG,
    }


@app.get("/dashboard/stats")
def get_main_dashboard_stats(db: Session = Depends(get_db)):
    from backend.app.api.routes.datasets import get_dashboard_stats
    return get_dashboard_stats(db=db)


# --------------------------------------------------------------------------- #
# Routers                                                                      #
# --------------------------------------------------------------------------- #

app.include_router(dataset_router)
app.include_router(generation_router)
app.include_router(image_router)
app.include_router(jobs_router)
app.include_router(kaggle_test_router)
app.include_router(deformation_router)
app.include_router(global_deformation_router)
app.include_router(languages_router)


# --------------------------------------------------------------------------- #
# Frontend                                                                     #
# Serve index.html at / and static assets from /static                        #
# --------------------------------------------------------------------------- #

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    if (FRONTEND_DIR / "test_gallery").exists():
        app.mount("/test_gallery", StaticFiles(directory=str(FRONTEND_DIR / "test_gallery")), name="test_gallery")


    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/", include_in_schema=False)
    def serve_frontend():
        target = FRONTEND_DIR / "index.html"
        if not target.exists():
            target = BASE_DIR / "frontend" / "index.html"
        return FileResponse(str(target))

    @app.get("/kaggle-test", include_in_schema=False)
    @app.get("/kaggle-test/", include_in_schema=False)
    def serve_kaggle_test_ui():
        target = FRONTEND_DIR / "kaggle_test.html"
        if not target.exists():
            target = BASE_DIR / "frontend" / "kaggle_test.html"
        return FileResponse(str(target))

    @app.get("/gallery", include_in_schema=False)
    @app.get("/gallery/", include_in_schema=False)
    def serve_gallery():
        target = FRONTEND_DIR / "index.html"
        if not target.exists():
            target = BASE_DIR / "frontend" / "index.html"
        return FileResponse(str(target))
