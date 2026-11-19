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
    datasets = get_all_datasets(db, skip=skip, limit=limit)
    from backend.app.core.status import map_job_status_to_dataset_status
    from backend.app.models.generation_job import GenerationJob
    from backend.app.services.timing_service import get_job_timing_summary

    res = []
    for d in datasets:
        latest_job = (
            db.query(GenerationJob)
            .filter(GenerationJob.dataset_id == d.id)
            .order_by(GenerationJob.id.desc())
            .first()
        )
        t_data = None
        if latest_job:
            eff_status = map_job_status_to_dataset_status(latest_job.status, stage=latest_job.current_stage).value
            if d.status != eff_status:
                d.status = eff_status
                db.commit()
                db.refresh(d)
            t_data = get_job_timing_summary(latest_job)
        item = DatasetResponse.model_validate(d)
        item.timing = t_data
        res.append(item)
    return res


# --------------------------------------------------------------------------- #
# GET /datasets/overview/stats                                                 #
# --------------------------------------------------------------------------- #

@router.get(
    "/overview/stats",
    summary="Get aggregated platform statistics for dashboard",
)
def get_dashboard_stats(
    db: Session = Depends(get_db),
):
    from sqlalchemy import distinct, func
    from backend.app.models.dataset import Dataset
    from backend.app.models.generated_image import GeneratedImage
    from backend.app.models.generation_job import GenerationJob
    from backend.app.core.config import settings

    total_datasets = db.query(func.count(Dataset.id)).scalar() or 0
    total_identities = db.query(func.count(distinct(Dataset.patient_id))).scalar() or total_datasets
    total_images = db.query(func.count(GeneratedImage.id)).scalar() or 0
    completed_datasets = (
        db.query(func.count(Dataset.id))
        .filter(Dataset.status == "COMPLETED")
        .scalar()
        or 0
    )
    total_deformations = (
        db.query(func.count(GeneratedImage.id))
        .filter(GeneratedImage.deformation_status == "COMPLETED")
        .scalar()
        or 0
    )

    recent_jobs = (
        db.query(GenerationJob)
        .order_by(GenerationJob.id.desc())
        .limit(5)
        .all()
    )
    jobs_data = [
        {
            "id": j.id,
            "dataset_id": j.dataset_id,
            "provider": j.provider,
            "status": j.status,
            "stage": getattr(j, "current_stage", None) or j.status,
            "generated_count": j.generated_count or 0,
            "created_at": j.created_at.isoformat() if j.created_at else None,
        }
        for j in recent_jobs
    ]

    recent_datasets = (
        db.query(Dataset)
        .order_by(Dataset.id.desc())
        .limit(6)
        .all()
    )
    from backend.app.core.status import map_job_status_to_dataset_status
    from backend.app.models.generation_job import GenerationJob

    datasets_data = []
    for d in recent_datasets:
        latest_job = (
            db.query(GenerationJob)
            .filter(GenerationJob.dataset_id == d.id)
            .order_by(GenerationJob.id.desc())
            .first()
        )
        eff_status = d.status
        if latest_job:
            eff_status = map_job_status_to_dataset_status(latest_job.status, stage=latest_job.current_stage).value
            if d.status != eff_status:
                d.status = eff_status
                db.commit()

        datasets_data.append({
            "id": d.id,
            "dataset_name": d.dataset_name,
            "patient_id": d.patient_id or f"P{d.id:02d}",
            "region": d.region,
            "population_group": d.population_group,
            "status": eff_status,
            "generated_count": d.generated_count or 0,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        })

    return {
        "total_identities": total_identities,
        "total_datasets": total_datasets,
        "total_generated_images": total_images,
        "completed_multiview_datasets": completed_datasets,
        "total_deformations": total_deformations,
        "available_morphologies": 4,
        "effective_provider": settings.effective_provider,
        "recent_jobs": jobs_data,
        "recent_datasets": datasets_data,
    }


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

    from backend.app.core.status import map_job_status_to_dataset_status
    from backend.app.models.generation_job import GenerationJob
    from backend.app.services.timing_service import get_job_timing_summary

    latest_job = (
        db.query(GenerationJob)
        .filter(GenerationJob.dataset_id == dataset.id)
        .order_by(GenerationJob.id.desc())
        .first()
    )
    timing_data = None
    if latest_job:
        eff_status = map_job_status_to_dataset_status(latest_job.status, stage=latest_job.current_stage).value
        if dataset.status != eff_status:
            dataset.status = eff_status
            db.commit()
            db.refresh(dataset)
        timing_data = get_job_timing_summary(latest_job)

    resp = DatasetResponse.model_validate(dataset)
    resp.timing = timing_data
    return resp


@router.get(
    "/{dataset_id}/timing",
    summary="Get generation and deformation timing metrics for a dataset",
)
def get_dataset_timing(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = get_dataset_by_id(db=db, dataset_id=dataset_id)
    if dataset is None:
        raise DatasetNotFoundException()

    from backend.app.models.generation_job import GenerationJob
    from backend.app.services.timing_service import get_job_timing_summary

    latest_job = (
        db.query(GenerationJob)
        .filter(GenerationJob.dataset_id == dataset.id)
        .order_by(GenerationJob.id.desc())
        .first()
    )
    if not latest_job:
        return {
            "dataset_id": dataset_id,
            "patient_id": dataset.patient_id or f"P{dataset_id:02d}",
            "status": dataset.status,
            "timing": None,
            "message": "No generation jobs recorded for this dataset.",
        }

    return {
        "dataset_id": dataset_id,
        "patient_id": dataset.patient_id or f"P{dataset_id:02d}",
        "status": dataset.status,
        "timing": get_job_timing_summary(latest_job),
    }


@router.get(
    "/{dataset_id}/images/{image_id}",
    summary="Serve an image belonging to a dataset",
)
def get_dataset_image_file(
    dataset_id: int,
    image_id: int,
    db: Session = Depends(get_db),
):
    from backend.app.models.generated_image import GeneratedImage
    image = (
        db.query(GeneratedImage)
        .filter(GeneratedImage.id == image_id, GeneratedImage.dataset_id == dataset_id)
        .first()
    )
    if not image or not Path(image.image_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(
        path=Path(image.image_path),
        media_type="image/png",
        filename=Path(image.image_path).name,
    )


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

    filename = f"dataset_{dataset_id}_export.zip"
    logger.info("Serving export ZIP for dataset %d: %s", dataset_id, zip_path)
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


# --------------------------------------------------------------------------- #
# GET /datasets/{dataset_id}/export                                            #
# --------------------------------------------------------------------------- #

@router.get(
    "/{dataset_id}/export",
    summary="Download an existing or freshly generated ZIP export",
    description=(
        "Builds or downloads the comprehensive ZIP export for the dataset. "
        "Returns 404 if the dataset does not exist or has not been completed."
    ),
)
def get_export(
    dataset_id: int,
    db: Session = Depends(get_db),
):
    dataset = get_dataset_by_id(db=db, dataset_id=dataset_id)
    if dataset is None:
        raise DatasetNotFoundException()

    if dataset.status not in (
        DatasetStatus.COMPLETED,
        DatasetStatus.COMPLETED_WITH_ERRORS,
    ):
        raise ExportNotFoundException()

    zip_path = build_export_zip(db, dataset)
    filename = f"dataset_{dataset_id}_export.zip"
    logger.info("Serving export ZIP for dataset %d: %s", dataset_id, zip_path)
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )