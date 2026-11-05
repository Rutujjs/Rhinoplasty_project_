"""
Isolated Kaggle GPU Integration Test Runner.

Executes ONE isolated test run of apac_face_generator.ipynb on Kaggle GPU
for kernel: rutujavaijanathpote/apac-face-generator.

Strict Boundaries:
- Zero PostgreSQL interaction (no database records created or modified).
- Production .env remains GENERATION_MODEL=MOCK.
- Isolated output directory: storage/isolated_test_artifacts
- Tests all 9 viewpoints:
  1. Front (0°)
  2. Left 20°
  3. Left 40°
  4. Left 60°
  5. Right 20°
  6. Right 40°
  7. Right 60°
  8. Basal
  9. Helicopter
- Validates batch_summary.json, image integrity, and backend validation pipeline.
- Generates 3x3 frontend gallery verification artifact.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings
from generation.interface import ProviderOutcome
from generation.kaggle_provider import KaggleBatchProvider, normalize_kernel_status
from generation.manifest import GenerationManifest, ManifestProviderConfig, build_prompt_from_attributes
from validation.pipeline import ValidationContext, validate_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("isolated_kaggle_test")

EXPECTED_VIEWPOINTS = [
    ("frontal", "Front (0°)", "Canonical frontal reference"),
    ("left_20", "Left 20°", "Nose → image-left"),
    ("left_40", "Left 40°", "Nose → image-left"),
    ("left_60", "Left 60°", "Nose → image-left"),
    ("right_20", "Right 20°", "Nose → image-right"),
    ("right_40", "Right 40°", "Nose → image-right"),
    ("right_60", "Right 60°", "Nose → image-right"),
    ("basal", "Basal", "Camera below nasal tip"),
    ("helicopter", "Helicopter", "Camera above forehead"),
]


def run_isolated_test():
    test_start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING ISOLATED KAGGLE GPU INTEGRATION TEST (OPTION A)")
    logger.info("=" * 70)

    # 1. Configuration Audit
    logger.info("Stage 0: Preflight configuration audit")
    logger.info("Kernel slug: %s", settings.KAGGLE_KERNEL_SLUG)
    logger.info("Manifest dataset: %s", settings.KAGGLE_MANIFEST_DATASET_SLUG)
    logger.info("Notebook path: %s", settings.KAGGLE_NOTEBOOK_PATH)
    logger.info("Production .env GENERATION_MODEL: %s (must be MOCK)", settings.GENERATION_MODEL)

    if "apac_face_generator.ipynb" not in str(settings.KAGGLE_NOTEBOOK_PATH):
        raise RuntimeError(f"Wrong notebook path configured: {settings.KAGGLE_NOTEBOOK_PATH}")

    # 2. Build Synthetic Patient & Canonical Prompt
    logger.info("\nStage 1: Building synthetic test patient & canonical prompt")
    patient_id = "P01"
    batch_id = "P01_isolated_test"
    seed = 20261034

    prompt = build_prompt_from_attributes(
        population_group="East Asian",
        region="Korea",
        age_min=25,
        age_max=30,
        sex_or_attribute="female",
        skin_tone="fair",
        pose="frontal",
        expression="neutral relaxed",
        lighting="soft clinical studio",
        medical_condition="Preoperative rhinoplasty nasal dorsum assessment",
        medical_severity="moderate",
    )
    logger.info("Generated canonical frontal prompt (%d chars):", len(prompt))
    logger.info("Prompt text:\n%s\n", prompt)

    # 3. Create Manifest
    logger.info("Stage 2: Creating manifest.json for isolated test")
    manifest = GenerationManifest(
        dataset_id=0,
        job_id=9999,
        batch_id=batch_id,
        requested_count=1,
        image_width=512,
        image_height=512,
        population_group="East Asian",
        region="Korea",
        age_min=25,
        age_max=30,
        sex_or_attribute="female",
        skin_tone="fair",
        pose="frontal",
        expression="neutral relaxed",
        lighting="soft clinical studio",
        medical_condition="Preoperative rhinoplasty nasal dorsum assessment",
        medical_severity="moderate",
        prompts=[prompt],
        seeds=[seed],
        provider_config=ManifestProviderConfig(
            provider="KAGGLE_BATCH",
            output_format="png",
        ),
    )

    test_dir = PROJECT_ROOT / "storage" / "isolated_test_artifacts"
    if test_dir.exists():
        logger.info("Purging old isolated test artifacts at %s", test_dir)
        shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    manifest_local_path = test_dir / "manifest.json"
    manifest.save(manifest_local_path)
    logger.info("Manifest saved to %s", manifest_local_path)

    # 4. Upload Manifest to Kaggle Dataset
    logger.info("\nStage 3: Uploading manifest to Kaggle dataset: %s", settings.KAGGLE_MANIFEST_DATASET_SLUG)
    provider = KaggleBatchProvider()
    prep_res = provider.prepare_batch(manifest)
    if not prep_res.succeeded:
        logger.error("Stage 3 FAILED: %s (detail: %s)", prep_res.message, prep_res.error_detail)
        sys.exit(1)
    logger.info("Manifest upload succeeded: %s", prep_res.message)

    # Wait for dataset processing to finish
    logger.info("Verifying dataset status...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    for wait_i in range(12):
        d_status = api.dataset_status(settings.KAGGLE_MANIFEST_DATASET_SLUG)
        logger.info("Dataset status: %s (check %d/12)", d_status, wait_i + 1)
        if d_status == "ready":
            break
        time.sleep(5)

    # 5. Submit Kernel Push
    logger.info("\nStage 4: Pushing kernel: %s with notebook apac_face_generator.ipynb", settings.KAGGLE_KERNEL_SLUG)
    push_start = time.time()
    sub_res = provider.submit_batch(manifest)
    if not sub_res.succeeded:
        logger.error("Stage 4 FAILED: %s (detail: %s)", sub_res.message, sub_res.error_detail)
        sys.exit(1)
    logger.info("Kernel pushed successfully. provider_run_id: %s", sub_res.provider_run_id)

    # 6. Poll Execution Status
    logger.info("\nStage 5: Polling Kaggle GPU execution...")
    poll_interval = 25  # seconds
    timeout = 3600      # 1 hour max
    consecutive_transient_errors = 0
    max_transient_errors = 10
    kernel_status = "unknown"

    logger.info("Waiting 30 seconds before first poll for container initialization...")
    time.sleep(30)

    gpu_execution_start = time.time()
    while True:
        elapsed = time.time() - gpu_execution_start
        if elapsed > timeout:
            logger.error("Stage 5 FAILED: Kernel timed out after %d seconds", elapsed)
            sys.exit(1)

        poll_res = provider.poll_status(settings.KAGGLE_KERNEL_SLUG)

        if poll_res.outcome == ProviderOutcome.PENDING:
            consecutive_transient_errors = 0
            kernel_status = poll_res.message
            logger.info("[Elapsed: %dm %02ds] Kaggle status: %s", int(elapsed // 60), int(elapsed % 60), poll_res.message)
            time.sleep(poll_interval)
            continue

        if poll_res.outcome == ProviderOutcome.SUCCESS:
            logger.info("=" * 50)
            logger.info("Kaggle kernel execution COMPLETED successfully! Total GPU time: %dm %02ds", int(elapsed // 60), int(elapsed % 60))
            logger.info("=" * 50)
            break

        if poll_res.outcome == ProviderOutcome.TRANSIENT_ERROR:
            consecutive_transient_errors += 1
            logger.warning("[Elapsed: %dm %02ds] Transient poll error (%d/%d): %s",
                           int(elapsed // 60), int(elapsed % 60), consecutive_transient_errors, max_transient_errors, poll_res.message)
            if consecutive_transient_errors >= max_transient_errors:
                logger.error("Stage 5 FAILED: Too many consecutive transient polling errors")
                sys.exit(1)
            time.sleep(poll_interval)
            continue

        # Permanent failure
        logger.error("Stage 5 FAILED: Kaggle kernel failed: %s (detail: %s)", poll_res.message, poll_res.error_detail)
        # Try downloading logs if possible
        try:
            provider.retrieve_artifacts(settings.KAGGLE_KERNEL_SLUG, str(test_dir), manifest)
        except Exception:
            pass
        sys.exit(1)

    gpu_duration_seconds = round(time.time() - gpu_execution_start, 2)

    # 7. Retrieve Artifacts
    logger.info("\nStage 6: Downloading Kaggle execution artifacts...")
    download_start = time.time()
    retrieve_res = provider.retrieve_artifacts(settings.KAGGLE_KERNEL_SLUG, str(test_dir), manifest)
    if not retrieve_res.succeeded:
        logger.error("Stage 6 FAILED: Could not retrieve artifacts: %s (detail: %s)", retrieve_res.message, retrieve_res.error_detail)
        sys.exit(1)
    logger.info("Artifacts retrieved in %.2fs: %s", time.time() - download_start, retrieve_res.message)

    # 8. Detailed Verification of Artifacts
    logger.info("\nStage 7: Verifying batch_summary.json and all 9 viewpoints")
    summary_candidates = list(test_dir.rglob("batch_summary.json"))
    if not summary_candidates:
        logger.error("Stage 7 FAILED: batch_summary.json not found in downloaded artifacts")
        sys.exit(1)

    summary_file = summary_candidates[0]
    logger.info("Found summary file at: %s", summary_file)
    summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
    logger.info("Summary content:\n%s", json.dumps(summary_data, indent=2))

    images_meta = summary_data.get("images", [])
    logger.info("Found %d image records in batch_summary.json", len(images_meta))

    # Map image records by viewpoint
    records_by_view = {}
    for entry in images_meta:
        v_key = entry.get("viewtype") or entry.get("view_key")
        if v_key:
            records_by_view[v_key] = entry

    missing_views = [k for k, _, _ in EXPECTED_VIEWPOINTS if k not in records_by_view]
    if missing_views:
        logger.error("Stage 7 FAILED: Missing viewpoints in batch_summary.json: %s", missing_views)
        sys.exit(1)
    logger.info("All 9 required viewpoints are present in batch_summary.json!")

    # 9. Validate each image on disk
    logger.info("\nStage 8: Running backend image validation on all 9 generated images")
    validated_images = []

    gallery_items = []
    target_patient_dir = PROJECT_ROOT / "storage" / "generated" / patient_id / "original"
    target_patient_dir.mkdir(parents=True, exist_ok=True)
    web_gallery_dir = PROJECT_ROOT / "frontend" / "test_gallery" / patient_id
    web_gallery_dir.mkdir(parents=True, exist_ok=True)

    for v_key, v_name, v_orient in EXPECTED_VIEWPOINTS:
        rec = records_by_view[v_key]
        img_name = rec.get("filename")
        matching_files = list(test_dir.rglob(img_name)) if img_name else []
        if not matching_files:
            # Fallback search for {patient_id}_{v_key}.png or *{v_key}*.png
            matching_files = list(test_dir.rglob(f"*{v_key}*.png"))

        if not matching_files:
            logger.error("Image file missing for viewpoint %s (%s)", v_key, img_name)
            sys.exit(1)

        img_path = matching_files[0]
        rec_w = rec.get("width")
        rec_h = rec.get("height")

        validation_ctx = ValidationContext(
            expected_width=rec_w if rec_w else None,
            expected_height=rec_h if rec_h else None,
            allowed_formats=["png", "jpg", "jpeg", "webp"],
            max_file_bytes=settings.MAX_IMAGE_FILE_BYTES,
            require_face_detection=False,
            enable_duplicate_detection=False,
        )

        v_res = validate_image(img_path, validation_ctx)

        logger.info(
            "View [%-11s | %-16s]: %dx%d, %d bytes, checksum: %s..., valid=%s",
            v_key, v_name, v_res.image_width, v_res.image_height, v_res.file_size_bytes,
            v_res.checksum_sha256[:12], v_res.accepted,
        )

        if not v_res.accepted:
            logger.error("Image validation failed for %s: %s", v_key, v_res.rejection_reasons)
            sys.exit(1)

        # Copy to storage/generated/P01/original/{v_key}.png
        storage_dest = target_patient_dir / f"{v_key}.png"
        shutil.copy2(img_path, storage_dest)

        # Copy to frontend/test_gallery/P01/{v_key}.png
        web_dest = web_gallery_dir / f"{v_key}.png"
        shutil.copy2(img_path, web_dest)

        validated_images.append({
            "viewpoint": v_key,
            "viewpoint_name": v_name,
            "orientation": v_orient,
            "filename": img_path.name,
            "path": str(storage_dest),
            "web_path": f"test_gallery/{patient_id}/{v_key}.png",
            "width": v_res.image_width,
            "height": v_res.image_height,
            "size_bytes": v_res.file_size_bytes,
            "sha256": v_res.checksum_sha256,
            "seed": rec.get("seed"),
        })

    # 10. Generate 3x3 Gallery HTML verification artifact
    logger.info("\nStage 9: Generating 3x3 gallery preview verification page")
    gallery_html_path = PROJECT_ROOT / "frontend" / "isolated_test_gallery.html"
    gallery_html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Isolated Kaggle GPU Test — Patient {patient_id} Multi-View Gallery</title>
  <style>
    body {{ background: #0b0d13; color: #e2e8f0; font-family: system-ui, sans-serif; padding: 2rem; margin: 0; }}
    h1 {{ color: #90cdf4; font-size: 1.5rem; margin-bottom: 0.5rem; }}
    .header-bar {{ background: #141824; border: 1px solid #232936; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; }}
    .badge {{ background: #22543d; color: #9ae6b4; font-size: 0.75rem; font-weight: 700; padding: 0.25rem 0.6rem; border-radius: 999px; text-transform: uppercase; }}
    .multiview-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; max-width: 1100px; margin: 0 auto; }}
    .card {{ background: #141824; border: 1px solid #232936; border-radius: 8px; padding: 0.75rem; display: flex; flex-direction: column; }}
    .card-head {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem; }}
    .card-title {{ font-weight: 700; font-size: 0.85rem; color: #edf2f7; }}
    .card-orient {{ font-size: 0.7rem; color: #718096; }}
    .img-wrap {{ width: 100%; aspect-ratio: 1; background: #050608; border: 1px solid #2d3748; border-radius: 6px; overflow: hidden; display: flex; align-items: center; justify-content: center; }}
    .img-wrap img {{ width: 100%; height: 100%; object-fit: contain; }}
    .card-foot {{ margin-top: 0.5rem; display: flex; justify-content: space-between; font-size: 0.72rem; color: #a0aec0; }}
  </style>
</head>
<body>
  <div class="header-bar">
    <div>
      <h1>Verified Kaggle FLUX.2 Klein 4B Multi-View Gallery</h1>
      <div style="font-size:0.8rem;color:#a0aec0">Patient: <strong>{patient_id}</strong> &bull; Kernel: <code>{settings.KAGGLE_KERNEL_SLUG}</code> &bull; Execution Time: <strong>{gpu_duration_seconds}s</strong></div>
    </div>
    <span class="badge">9/9 Viewpoints Verified</span>
  </div>
  <div class="multiview-grid">
"""
    for v in validated_images:
        gallery_html_content += f"""    <div class="card">
      <div class="card-head">
        <div>
          <div class="card-title">{v['viewpoint_name']}</div>
          <div class="card-orient">{v['orientation']}</div>
        </div>
        <span class="badge">ACCEPTED</span>
      </div>
      <div class="img-wrap">
        <img src="{v['web_path']}" alt="{v['viewpoint_name']}">
      </div>
      <div class="card-foot">
        <span>{v['width']}×{v['height']} px</span>
        <span>Seed: {v['seed']}</span>
      </div>
    </div>
"""
    gallery_html_content += """  </div>
</body>
</html>
"""
    gallery_html_path.write_text(gallery_html_content, encoding="utf-8")
    logger.info("3x3 Gallery verification page created at: %s", gallery_html_path)

    # 11. Compile Final Test Report JSON
    total_test_duration = round(time.time() - test_start_time, 2)
    meta_json_files = list(test_dir.rglob("metadata.json"))
    patient_meta = {}
    if meta_json_files:
        try:
            patient_meta = json.loads(meta_json_files[0].read_text(encoding="utf-8"))
            logger.info("Parsed patient metadata.json successfully")
        except Exception as e:
            logger.warning("Could not parse metadata.json: %s", e)

    report = {
        "status": "SUCCESS",
        "kernel_slug": settings.KAGGLE_KERNEL_SLUG,
        "notebook": "apac_face_generator.ipynb",
        "patient_id": patient_id,
        "batch_id": batch_id,
        "seed": seed,
        "gpu_duration_seconds": gpu_duration_seconds,
        "total_test_duration_seconds": total_test_duration,
        "viewpoints_generated": len(validated_images),
        "viewpoints": validated_images,
        "patient_metadata": patient_meta,
        "summary_metadata": summary_data,
        "gallery_html": str(gallery_html_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = PROJECT_ROOT / "storage" / "isolated_test_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved final test report to: %s", report_path)

    logger.info("=" * 70)
    logger.info("ISOLATED KAGGLE GPU INTEGRATION TEST COMPLETED SUCCESSFULLY!")
    logger.info("All 9 viewpoints generated, validated, and verified.")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_isolated_test()
