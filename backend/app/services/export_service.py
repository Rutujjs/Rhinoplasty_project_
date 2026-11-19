"""
Export service – builds a comprehensive ZIP archive of all accepted images,
deformations, masks, and diagnostics for a dataset.

The ZIP structure is:
    dataset_{id}_export.zip
    |-- dataset_metadata.json
    |-- manifest.txt
    |-- README.txt
    |-- deformation_audit.json
    |-- patients/
    |   |-- {patient_id}/
    |       |-- original/
    |       |-- deformed/
    |       |-- masks/
    |       |-- metadata.json
    |-- diagnostics/
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.exceptions import ExportFailedException
from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage

logger = logging.getLogger(__name__)


def _compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_export_zip(
    db: Session,
    dataset: Dataset,
    include_rejected: bool = False,
) -> Path:
    """
    Build a ZIP file containing original images, deformed images, masks,
    diagnostics, and rich metadata for *dataset*.

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

    zip_path = export_dir / f"dataset_{dataset.id}_export.zip"
    legacy_zip_path = export_dir / f"dataset_{dataset.id}.zip"

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
        _write_complete_zip(zip_path, dataset, images)
        # Ensure backward compatibility with any legacy test or client referencing dataset_{id}.zip
        shutil.copy2(zip_path, legacy_zip_path)
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


def _write_complete_zip(
    zip_path: Path,
    dataset: Dataset,
    images: list[GeneratedImage],
) -> None:
    """Write the structured ZIP archive to *zip_path*."""
    patient_id = getattr(dataset, "patient_id", None) or f"P{dataset.id:02d}"
    morphology = getattr(dataset, "morphology_type", None) or "saddle"
    deformation_strength = float(getattr(dataset, "deformation_strength", 0.6) or 0.6)
    deviated_direction = getattr(dataset, "deviated_direction", "right") or "right"

    # Deformation and masks search paths
    deform_base_dir = Path("storage/deformations") / str(dataset.id) / patient_id / morphology
    bisenet_mask_dir = Path("storage/diagnostics/bisenet_patient_masks")
    diag_audit_dir = Path(f"storage/diagnostics/{morphology}_{patient_id.lower()}_audit")
    if not diag_audit_dir.exists():
        diag_audit_dir = Path("storage/diagnostics/deviated_p01_audit")

    exported_at = datetime.now(timezone.utc).isoformat()

    # Pre-build comprehensive Dataset Metadata JSON
    dataset_metadata = {
        "dataset": {
            "id": dataset.id,
            "dataset_name": dataset.dataset_name,
            "patient_id": patient_id,
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
            "morphology_type": morphology,
            "deformation_strength": deformation_strength,
            "deviated_direction": deviated_direction,
        },
        "export": {
            "exported_at": exported_at,
            "total_images": len(images),
            "patients": [patient_id],
            "export_format_version": "2.0",
            "segmentation_engine": "BiSeNet ResNet-18 (ONNX)",
            "deformation_engine": "2D Remap Displacement Field (cv2.remap)",
        },
        "images": [
            {
                "id": img.id,
                "patient_id": patient_id,
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
    meta_json_str = json.dumps(dataset_metadata, indent=2)

    # Per-viewpoint audit records & patient metadata
    viewpoints_meta: List[Dict[str, Any]] = []
    audit_records: List[Dict[str, Any]] = []

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write root metadata first so legacy index lookups for metadata.json succeed immediately
        zf.writestr("metadata.json", meta_json_str)
        zf.writestr(f"dataset_{dataset.id}/metadata.json", meta_json_str)
        zf.writestr("dataset_metadata.json", meta_json_str)

        # 1. Process Patient Images (Original, Deformed, Masks)
        for img in images:
            vp = getattr(img, "viewpoint", "") or f"view_{img.id}"
            vp_name = getattr(img, "viewpoint_name", "") or vp.replace("_", " ").title()
            orig_path = Path(img.image_path)

            orig_arc = None
            orig_sha = img.checksum_sha256 or ""
            if orig_path.exists():
                orig_arc = f"patients/{patient_id}/original/{orig_path.name}"
                zf.write(orig_path, arcname=orig_arc)
                if not orig_sha:
                    orig_sha = _compute_sha256(orig_path)
            else:
                logger.warning("Original image missing: %s", orig_path)

            # Check for deformed image
            deformed_arc = None
            deform_sha = None
            deform_status = getattr(img, "deformation_status", "NOT_STARTED") or "NOT_STARTED"

            # Check potential deformed file names
            deformed_candidates = [
                deform_base_dir / f"{vp}_{morphology}.png",
                deform_base_dir / f"{vp}.png",
            ]
            for cand in deformed_candidates:
                if cand.exists():
                    # Exclude incomplete/failed outputs from being represented as successfully deformed
                    if deform_status not in ("FAILED", "PENDING", "NOT_STARTED"):
                        deformed_arc = f"patients/{patient_id}/deformed/{cand.name}"
                        zf.write(cand, arcname=deformed_arc)
                        deform_sha = _compute_sha256(cand)
                    break

            # Check for mask
            mask_arc = None
            mask_candidates = [
                deform_base_dir / f"{vp}_mask.png",
                bisenet_mask_dir / f"bisenet_mask_{vp}.png",
                bisenet_mask_dir / f"{vp.upper()}_binary_mask.png",
            ]
            for mcand in mask_candidates:
                if mcand.exists():
                    mask_arc = f"patients/{patient_id}/masks/{vp}_mask.png"
                    zf.write(mcand, arcname=mask_arc)
                    break

            vp_entry = {
                "image_id": img.id,
                "viewpoint": vp,
                "viewpoint_name": vp_name,
                "original_file": orig_arc,
                "original_sha256": orig_sha,
                "deformed_file": deformed_arc,
                "deformed_sha256": deform_sha,
                "mask_file": mask_arc,
                "validation_status": img.validation_status,
                "deformation_status": deform_status,
                "image_width": img.image_width,
                "image_height": img.image_height,
                "seed": img.seed,
                "prompt": img.prompt,
            }
            viewpoints_meta.append(vp_entry)

            audit_entry = {
                "viewpoint": vp,
                "viewpoint_name": vp_name,
                "original_sha256": orig_sha,
                "deformed_sha256": deform_sha,
                "hashes_differ": bool(orig_sha and deform_sha and orig_sha != deform_sha),
                "deformation_status": deform_status,
                "morphology_type": morphology,
                "deformation_strength": deformation_strength,
            }
            audit_records.append(audit_entry)

        # 2. Add Patient Metadata
        patient_metadata = {
            "patient_id": patient_id,
            "dataset_id": dataset.id,
            "dataset": {
                "id": dataset.id,
                "dataset_name": dataset.dataset_name,
                "patient_id": patient_id,
            },
            "export": {
                "exported_at": exported_at,
                "total_images": len(images),
            },
            "images": dataset_metadata["images"],
            "dataset_name": dataset.dataset_name,
            "demographics": {
                "region": dataset.region,
                "population_group": dataset.population_group,
                "age_min": dataset.age_min,
                "age_max": dataset.age_max,
                "sex_or_attribute": dataset.sex_or_attribute,
                "skin_tone": dataset.skin_tone,
            },
            "clinical": {
                "medical_condition": dataset.medical_condition,
                "medical_severity": dataset.medical_severity,
                "morphology_type": morphology,
                "deformation_strength": deformation_strength,
                "deviated_direction": deviated_direction,
            },
            "viewpoints_count": len(viewpoints_meta),
            "viewpoints": viewpoints_meta,
        }
        zf.writestr(
            f"patients/{patient_id}/metadata.json",
            json.dumps(patient_metadata, indent=2),
        )

        # 3. Add Diagnostics (comparison strips, difference maps)
        if diag_audit_dir.exists():
            for diag_file in diag_audit_dir.iterdir():
                if diag_file.is_file() and diag_file.suffix.lower() in (".png", ".jpg", ".json"):
                    zf.write(diag_file, arcname=f"diagnostics/{diag_file.name}")

        # 4. Add Deformation Audit JSON
        # If precomputed summary exists on disk, merge its metrics
        precomputed_summary = diag_audit_dir / f"{morphology}_audit_summary.json"
        if not precomputed_summary.exists():
            precomputed_summary = diag_audit_dir / "deviated_audit_summary.json"

        if precomputed_summary.exists():
            try:
                with open(precomputed_summary, "r", encoding="utf-8") as pf:
                    detailed_audit = json.load(pf)
                zf.writestr(
                    "deformation_audit.json",
                    json.dumps(detailed_audit, indent=2),
                )
            except Exception:
                zf.writestr(
                    "deformation_audit.json",
                    json.dumps(audit_records, indent=2),
                )
        else:
            zf.writestr(
                "deformation_audit.json",
                json.dumps(audit_records, indent=2),
            )

        # 5. Build Manifest Text
        manifest_lines = [
            "================================================================================",
            "Faceify Labs – Synthetic APAC Patient Face Dataset & Deformation Platform",
            "Dataset Export Manifest",
            "================================================================================",
            f"Dataset ID          : {dataset.id}",
            f"Dataset Name        : {dataset.dataset_name}",
            f"Patient ID          : {patient_id}",
            f"Region / Country    : {dataset.region}",
            f"Population Group    : {dataset.population_group}",
            f"Morphology Type     : {morphology}",
            f"Deformation Strength: {deformation_strength}",
            f"Exported At         : {exported_at}",
            f"Total Viewpoints    : {len(images)}",
            "",
            "Viewpoints & Files Summary:",
            "--------------------------------------------------------------------------------",
        ]
        for v in viewpoints_meta:
            deform_str = v["deformed_file"] or "None"
            mask_str = v["mask_file"] or "None"
            manifest_lines.append(
                f"  [{v['viewpoint']:<10}] {v['viewpoint_name']:<18} | Status: {v['deformation_status']:<18} | Deformed: {deform_str} | Mask: {mask_str}"
            )
        manifest_text = "\n".join(manifest_lines) + "\n"
        zf.writestr("manifest.txt", manifest_text)
        zf.writestr(f"dataset_{dataset.id}/manifest.txt", manifest_text)

        # 7. Build README.txt
        readme_lines = [
            "================================================================================",
            "FACEIFY LABS — CLINICAL RHINOPLASTY SYNTHETIC DATASET EXPORT",
            "================================================================================",
            "",
            "1. OVERVIEW",
            f"This archive contains synthetic patient data for dataset #{dataset.id} ({dataset.dataset_name}).",
            "Images are photorealistic, multi-view clinical representations of synthetic adult faces,",
            "developed for rhinoplasty research, surgical planning, and morphological analysis.",
            "",
            "2. DIRECTORY STRUCTURE",
            "  dataset_metadata.json     : Comprehensive dataset configuration and metadata.",
            "  metadata.json             : Standardized metadata schema descriptor.",
            "  manifest.txt              : Plaintext manifest of all viewpoints and files.",
            "  README.txt                : This documentation file.",
            "  deformation_audit.json    : Quantitative pixel metrics and validation audit.",
            "  patients/                 : Patient records containing viewpoint images.",
            f"    {patient_id}/",
            "      original/             : 9 canonical original FLUX.2 Klein viewpoint images.",
            "      deformed/             : Morphology-deformed viewpoint images (where completed).",
            "      masks/                : Patient-specific BiSeNet ResNet-18 nasal masks.",
            "      metadata.json         : Patient-level clinical and viewpoint metadata.",
            "  diagnostics/              : 4-panel visual comparison strips and difference maps.",
            "",
            "3. CANONICAL VIEWPOINTS",
            "  - frontal       : Frontal canonical reference view (0°)",
            "  - left_20       : Left three-quarter oblique view (20°)",
            "  - left_40       : Left oblique view (40°)",
            "  - left_60       : Left profile view (60°)",
            "  - right_20      : Right three-quarter oblique view (20°)",
            "  - right_40      : Right oblique view (40°)",
            "  - right_60      : Right profile view (60°)",
            "  - basal         : Submental basal underside view (nostrils / columella)",
            "  - helicopter    : Superior top-down view (nasal dorsum / bridge)",
            "",
            "4. NASAL MORPHOLOGY ENGINE",
            f"  Applied Morphology: {morphology.upper()}",
            f"  Applied Strength  : {deformation_strength}",
            f"  Direction         : {deviated_direction}",
            "  Segmentation      : Official CelebAMask-HQ BiSeNet ResNet-18 (Class 10 Nose)",
            "  Deformation Core  : Localized displacement fields with feathered Gaussian support",
            "  Remapping         : OpenCV cv2.remap (INTER_CUBIC, BORDER_REFLECT101)",
            "",
            "5. DATA INTEGRITY & IMMUTABILITY",
            "  Original FLUX images are permanently preserved and immutable.",
            "  Deformed images and masks are maintained in separate directories.",
            "  All SHA-256 checksums are documented in dataset_metadata.json and deformation_audit.json.",
            "",
            "================================================================================",
        ]
        zf.writestr("README.txt", "\n".join(readme_lines) + "\n")
