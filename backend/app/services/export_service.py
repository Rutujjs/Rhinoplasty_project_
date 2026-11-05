"""
Export service – builds a ZIP archive of all accepted images for a dataset.

The ZIP structure is:
    dataset_{id}/
        images/
            <image_filename>  (one file per accepted image)
        metadata.json         (dataset config + per-image metadata)
        manifest.txt          (summary stats)

Usage
-----
    from backend.app.services.export_service import build_export_zip

    zip_path = build_export_zip(db, dataset)

Returns the Path of the created ZIP file, or raises ExportFailedException.
"""

from __future__ import annotations

import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.exceptions import ExportFailedException
from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage

logger = logging.getLogger(__name__)


def build_export_zip(
    db: Session,
    dataset: Dataset,
    include_rejected: bool = False,
) -> Path:
    """
    Build a ZIP file containing all accepted (and optionally rejected) images
    for *dataset*.

    Parameters
    ----------
    db               : active database session
    dataset          : Dataset ORM instance
    include_rejected : also include images with validation_status REJECTED

    Returns
    -------
    Path to the created ZIP file (inside EXPORT_DIR).

    Raises
    ------
    ExportFailedException – if no images are available or an I/O error occurs.
    """
    export_dir = Path(settings.EXPORT_DIR)
    export_dir.mkdir(parents=True, exist_ok=True)

    zip_path = export_dir / f"dataset_{dataset.id}.zip"

    # Collect images.
    statuses = ["ACCEPTED"]
    if include_rejected:
        statuses.append("REJECTED")

    images: list[GeneratedImage] = (
        db.query(GeneratedImage)
        .filter(
            GeneratedImage.dataset_id == dataset.id,
            GeneratedImage.validation_status.in_(statuses),
        )
        .order_by(GeneratedImage.id.asc())
        .all()
    )

    if not images:
        raise ExportFailedException(
            f"Dataset {dataset.id} has no exportable images "
            f"(validation_status in {statuses})."
        )

    try:
        _write_zip(zip_path, dataset, images)
    except Exception as exc:
        logger.exception("Failed to write ZIP for dataset %d", dataset.id)
        raise ExportFailedException(str(exc)) from exc

    logger.info(
        "Export ZIP created for dataset %d: %s (%d images, %.1f KB)",
        dataset.id,
        zip_path,
        len(images),
        zip_path.stat().st_size / 1024,
    )
    return zip_path


def _write_zip(
    zip_path: Path,
    dataset: Dataset,
    images: list[GeneratedImage],
) -> None:
    """Write the ZIP archive to *zip_path*."""
    folder = f"dataset_{dataset.id}"

    # Build metadata payload.
    metadata = {
        "dataset": {
            "id": dataset.id,
            "dataset_name": dataset.dataset_name,
            "patient_id": getattr(dataset, "patient_id", None) or f"P{dataset.id:02d}",
            "region": dataset.region,
            "population_group": dataset.population_group,
            "age_min": dataset.age_min,
            "age_max": dataset.age_max,
            "sex_or_attribute": dataset.sex_or_attribute,
            "skin_tone": dataset.skin_tone,
            "pose": dataset.pose,
            "expression": dataset.expression,
            "medical_condition": dataset.medical_condition,
            "medical_severity": dataset.medical_severity,
            "num_images": dataset.num_images,
            "resolution": dataset.resolution,
            "status": str(dataset.status),
        },
        "export": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_images": len(images),
        },
        "images": [
            {
                "id": img.id,
                "patient_id": getattr(img, "patient_id", None),
                "viewpoint": getattr(img, "viewpoint", None),
                "viewpoint_name": getattr(img, "viewpoint_name", None),
                "filename": Path(img.image_path).name,
                "generation_model": img.generation_model,
                "seed": img.seed,
                "prompt": img.prompt,
                "validation_status": img.validation_status,
                "deformation_status": getattr(img, "deformation_status", "NOT_STARTED"),
                "image_width": img.image_width,
                "image_height": img.image_height,
                "file_size_bytes": img.file_size_bytes,
                "checksum_sha256": img.checksum_sha256,
                "created_at": img.created_at.isoformat() if img.created_at else None,
            }
            for img in images
        ],
    }

    manifest_lines = [
        f"Synthetic APAC Patient Face Dataset – Export Manifest",
        f"Dataset ID   : {dataset.id}",
        f"Dataset name : {dataset.dataset_name}",
        f"Exported at  : {metadata['export']['exported_at']}",
        f"Total images : {len(images)}",
        "",
        "Images",
        "------",
    ] + [
        f"  {Path(img.image_path).name}  (id={img.id}, viewpoint={getattr(img, 'viewpoint', 'N/A')}, seed={img.seed})"
        for img in images
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write metadata.json
        zf.writestr(
            f"{folder}/metadata.json",
            json.dumps(metadata, indent=2),
        )

        # Write manifest.txt
        zf.writestr(
            f"{folder}/manifest.txt",
            "\n".join(manifest_lines) + "\n",
        )

        # Write image files.
        missing = 0
        for img in images:
            img_path = Path(img.image_path)
            if img_path.exists():
                arcname = f"{folder}/images/{img_path.name}"
                zf.write(img_path, arcname=arcname)
            else:
                logger.warning(
                    "Image file missing, skipping: %s (image_id=%d)",
                    img.image_path,
                    img.id,
                )
                missing += 1

        if missing == len(images):
            raise ExportFailedException(
                f"All {missing} image files are missing from disk."
            )

        if missing:
            logger.warning(
                "%d/%d image files were missing for dataset %d export",
                missing,
                len(images),
                dataset.id,
            )
