"""
Dataset CRUD routes.

Endpoints
---------
POST   /datasets/             – create a dataset
GET    /datasets/             – list datasets (with pagination)
GET    /datasets/{id}         – retrieve a dataset
DELETE /datasets/{id}         – delete a dataset (state-dependent)
POST   /datasets/{id}/export  – request a ZIP export
GET    /datasets/{id}/export  – get export status / download URL
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.core.exceptions import (
    DatasetDeleteForbiddenException,
    DatasetNotFoundException,
    ExportFailedException,
    ExportNotFoundException,
)
from backend.app.core.config import settings
from backend.app.core.status import DatasetStatus
from backend.app.database.session import get_db
from backend.app.schemas.dataset import DatasetCreate, DatasetResponse
from backend.app.schemas.job import ExportRequest, ExportStatusResponse
from backend.app.services.dataset_service import (
    create_dataset,
    delete_dataset,
    get_all_datasets,
    get_dataset_by_id,
)
from backend.app.services.export_service import build_export_zip

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)


# --------------------------------------------------------------------------- #
# POST /datasets/                                                               #
# --------------------------------------------------------------------------- #

@router.post(
    "/",
    response_model=DatasetResponse,
    status_code=200,
    summary="Create a new dataset configuration",
)
def create_new_dataset(
    dataset_data: DatasetCreate,
    db: Session = Depends(get_db),
):
    return create_dataset(db=db, dataset_data=dataset_data)


# --------------------------------------------------------------------------- #
# GET /datasets/                                                                #
# --------------------------------------------------------------------------- #

@router.get(
    "/",
    response_model=list[DatasetResponse],
    summary="List all datasets",
)
def list_datasets(
    db: Session = Depends(get_db),
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum records to return"),
):
    return get_all_datasets(db, skip=skip, limit=limit)


# --------------------------------------------------------------------------- #
# GET /datasets/{dataset_id}                                                   #
# --------------------------------------------------------------------------- #

@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Retrieve a dataset",
)
def retrieve_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = get_dataset_by_id(db=db, dataset_id=dataset_id)
    if dataset is None:
        raise DatasetNotFoundException()
    return dataset


# --------------------------------------------------------------------------- #
# DELETE /datasets/{dataset_id}                                                #
# --------------------------------------------------------------------------- #

@router.delete(
    "/{dataset_id}",
    status_code=204,
    summary="Delete a dataset",
    description=(
        "A dataset may only be deleted when it is in CREATED or FAILED status. "
        "Datasets that are currently GENERATING cannot be deleted."
    ),
)
def delete_dataset_endpoint(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = get_dataset_by_id(db=db, dataset_id=dataset_id)
    if dataset is None:
        raise DatasetNotFoundException()

    # Guard: do not delete a dataset that is actively generating.
    if dataset.status == DatasetStatus.GENERATING:
        raise DatasetDeleteForbiddenException(
            "Dataset cannot be deleted while generation is in progress. "
            "Cancel the active job first."
        )

    delete_dataset(db=db, dataset=dataset)
    # 204 No Content – return nothing


# --------------------------------------------------------------------------- #
# POST /datasets/{dataset_id}/export                                           #
# --------------------------------------------------------------------------- #

@router.post(
    "/{dataset_id}/export",
    summary="Build and download a ZIP export of the dataset",
    description=(
        "Builds a ZIP archive of all accepted images for the dataset and "
        "returns the file immediately.  Only completed datasets can be "
        "exported.  Pass include_rejected=true to include rejected images."
    ),
)
def request_export(
    dataset_id: int,
    export_request: ExportRequest = ExportRequest(),
    db: Session = Depends(get_db),
):
    dataset = get_dataset_by_id(db=db, dataset_id=dataset_id)
    if dataset is None:
        raise DatasetNotFoundException()

    if dataset.status not in (
        DatasetStatus.COMPLETED,
        DatasetStatus.COMPLETED_WITH_ERRORS,
    ):
        raise ExportFailedException(
            "Only completed datasets can be exported. "
            f"Current status: {dataset.status}"
        )

    zip_path = build_export_zip(
        db,
        dataset,
        include_rejected=export_request.include_rejected,
    )

    logger.info("Serving export ZIP for dataset %d: %s", dataset_id, zip_path)
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=zip_path.name,
    )


# --------------------------------------------------------------------------- #
# GET /datasets/{dataset_id}/export                                            #
# --------------------------------------------------------------------------- #

@router.get(
    "/{dataset_id}/export",
    summary="Download an existing ZIP export",
    description=(
        "Downloads a previously built ZIP export.  Returns 404 if no export "
        "exists yet – call POST /datasets/{id}/export to build one."
    ),
)
def get_export(
    dataset_id: int,
    db: Session = Depends(get_db),
):

    dataset = get_dataset_by_id(db=db, dataset_id=dataset_id)
    if dataset is None:
        raise DatasetNotFoundException()

    export_dir = Path(settings.EXPORT_DIR)
    zip_path = export_dir / f"dataset_{dataset_id}.zip"

    if zip_path.exists():
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=zip_path.name,
        )

    raise ExportNotFoundException()