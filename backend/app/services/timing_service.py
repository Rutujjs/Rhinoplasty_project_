"""
Generation & Deformation Timing Service.

Tracks exact timestamps, durations, statuses, errors, and per-viewpoint metrics
for all 11 lifecycle stages of synthetic face generation and morphology deformation:

1. Queue waiting time
2. Manifest preparation time
3. Kaggle submission time
4. FLUX.2 Klein generation time
5. Artifact download time
6. Image validation time
7. BiSeNet segmentation time
8. Binary mask generation time
9. Deformation processing time
10. Database and file persistence time
11. Complete end-to-end elapsed time
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Typical benchmark durations (in seconds) for estimation when a stage is active:
STAGE_ESTIMATED_DURATIONS = {
    "manifest_preparation": 5.0,
    "kaggle_submission": 3.0,
    "flux_generation": 360.0,
    "artifact_download": 35.0,
    "image_validation": 10.0,
    "segmentation": 0.5,
    "binary_mask": 0.3,
    "deformation": 15.0,
    "persistence": 2.0,
}

ALL_STAGES_ORDERED = [
    "queue_waiting",
    "manifest_preparation",
    "kaggle_submission",
    "flux_generation",
    "artifact_download",
    "image_validation",
    "segmentation",
    "binary_mask",
    "deformation",
    "persistence",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _get_timing_dict(job) -> dict:
    raw = getattr(job, "stage_timing", None)
    if not raw:
        return {
            "stages": {},
            "per_view_timing": {},
            "total_elapsed_sec": 0.0,
            "estimated_remaining_sec": None,
            "completed_viewpoints": 0,
        }
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw)
        if "stages" not in data:
            data["stages"] = {}
        if "per_view_timing" not in data:
            data["per_view_timing"] = {}
        return data
    except Exception:
        return {
            "stages": {},
            "per_view_timing": {},
            "total_elapsed_sec": 0.0,
            "estimated_remaining_sec": None,
            "completed_viewpoints": 0,
        }


def _save_timing_dict(job, data: dict) -> None:
    job.stage_timing = json.dumps(data)


def record_stage_start(job, stage_name: str) -> None:
    """Record start timestamp for a processing stage."""
    data = _get_timing_dict(job)
    stages = data.setdefault("stages", {})

    now = _now_iso()
    stage_entry = stages.get(stage_name, {})
    if "start" not in stage_entry or stage_entry.get("status") != "COMPLETED":
        stage_entry["start"] = now
        stage_entry["status"] = "IN_PROGRESS"
        stage_entry["error"] = None
        stages[stage_name] = stage_entry

    _save_timing_dict(job, data)


def record_stage_end(
    job,
    stage_name: str,
    status: str = "COMPLETED",
    error: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Record completion timestamp and calculate duration for a stage."""
    data = _get_timing_dict(job)
    stages = data.setdefault("stages", {})

    now_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    now = now_dt.isoformat()

    stage_entry = stages.get(stage_name, {})
    start_str = stage_entry.get("start")
    duration = 0.0

    if start_str:
        try:
            start_dt = datetime.fromisoformat(start_str)
            duration = max(0.0, round((now_dt - start_dt).total_seconds(), 2))
        except Exception:
            duration = 0.0

    stage_entry["end"] = now
    stage_entry["duration_sec"] = duration
    stage_entry["status"] = status
    stage_entry["error"] = error
    if extra_data:
        stage_entry.update(extra_data)

    stages[stage_name] = stage_entry

    # Calculate overall total elapsed time (from job.started_at or first active stage)
    if job.started_at:
        try:
            total_sec = (now_dt - job.started_at).total_seconds()
            data["total_elapsed_sec"] = max(0.0, round(total_sec, 2))
        except Exception:
            pass

    _save_timing_dict(job, data)
    logger.info("⏱️ [TIMING] Job #%s | Stage '%s': %s (Duration: %.2fs)", getattr(job, "id", "?"), stage_name, status, duration)


def record_view_timing(
    job,
    viewpoint: str,
    seg_sec: float,
    def_sec: float,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Record per-viewpoint segmentation, deformation, and total time."""
    data = _get_timing_dict(job)
    pvt = data.setdefault("per_view_timing", {})
    pvt[viewpoint] = {
        "viewpoint": viewpoint,
        "segmentation_sec": round(float(seg_sec), 3),
        "seg_sec": round(float(seg_sec), 3),
        "deformation_sec": round(float(def_sec), 3),
        "def_sec": round(float(def_sec), 3),
        "total_sec": round(float(seg_sec + def_sec), 3),
        "metrics": metrics or {},
    }
    data["completed_viewpoints"] = len(pvt)
    _save_timing_dict(job, data)
    logger.info(
        "⏱️ [TIMING] Job #%s | Viewpoint '%s': seg=%.3fs, deform=%.3fs, total=%.3fs",
        getattr(job, "id", "?"),
        viewpoint,
        seg_sec,
        def_sec,
        round(seg_sec + def_sec, 3),
    )


def estimate_remaining_seconds(job) -> Optional[float]:
    """
    Estimate remaining processing time in seconds based on current stage and elapsed time.
    Returns None if waiting in queue or already finished.
    """
    from backend.app.core.status import JobStatus
    try:
        js = JobStatus(job.status)
    except Exception:
        return None

    if js.is_terminal:
        return 0.0
    if js == JobStatus.QUEUED:
        return None

    data = _get_timing_dict(job)
    stages = data.get("stages", {})

    # Determine remaining stages
    remaining_sec = 0.0
    stage_map = {
        JobStatus.PREPARING: ["manifest_preparation", "kaggle_submission", "flux_generation", "artifact_download", "image_validation", "deformation", "persistence"],
        JobStatus.SUBMITTED: ["kaggle_submission", "flux_generation", "artifact_download", "image_validation", "deformation", "persistence"],
        JobStatus.RUNNING: ["flux_generation", "artifact_download", "image_validation", "deformation", "persistence"],
        JobStatus.DOWNLOADING: ["artifact_download", "image_validation", "deformation", "persistence"],
        JobStatus.VALIDATING: ["image_validation", "deformation", "persistence"],
        JobStatus.PACKAGING: ["deformation", "persistence"],
    }

    stages_left = stage_map.get(js, [])
    now_dt = datetime.now(timezone.utc).replace(tzinfo=None)

    for stg in stages_left:
        stg_entry = stages.get(stg, {})
        default_dur = STAGE_ESTIMATED_DURATIONS.get(stg, 5.0)

        if stg_entry.get("status") == "IN_PROGRESS" and stg_entry.get("start"):
            try:
                stg_start = datetime.fromisoformat(stg_entry["start"])
                elapsed = (now_dt - stg_start).total_seconds()
                remaining_sec += max(1.0, default_dur - elapsed)
            except Exception:
                remaining_sec += default_dur
        elif stg_entry.get("status") != "COMPLETED":
            remaining_sec += default_dur

    return max(1.0, round(remaining_sec, 1))


def get_job_timing_summary(job) -> Dict[str, Any]:
    """Build a comprehensive, human-readable timing payload for frontend rendering."""
    data = _get_timing_dict(job)
    stages = data.get("stages", {})
    now_dt = datetime.now(timezone.utc).replace(tzinfo=None)

    # Compute queue duration
    queue_sec = 0.0
    if job.created_at:
        end_ref = job.started_at or job.finished_at or (now_dt if job.status == "QUEUED" else None)
        if end_ref:
            queue_sec = max(0.0, round((end_ref - job.created_at).total_seconds(), 2))

    # Compute total elapsed duration
    total_elapsed = data.get("total_elapsed_sec", 0.0)
    if job.started_at:
        end_ref = job.finished_at or now_dt
        total_elapsed = max(0.0, round((end_ref - job.started_at).total_seconds(), 2))

    # Calculate FLUX-only duration vs Deformation-only duration
    flux_sec = stages.get("flux_generation", {}).get("duration_sec", 0.0)
    def_sec = stages.get("deformation", {}).get("duration_sec", 0.0)
    seg_sec = stages.get("segmentation", {}).get("duration_sec", 0.0)

    # If FLUX is currently running, compute active duration
    if job.status == "RUNNING" and stages.get("flux_generation", {}).get("start"):
        try:
            st = datetime.fromisoformat(stages["flux_generation"]["start"])
            flux_sec = max(0.0, round((now_dt - st).total_seconds(), 2))
        except Exception:
            pass

    return {
        "job_id": job.id,
        "dataset_id": job.dataset_id,
        "status": job.status,
        "current_stage": job.current_stage,
        "queue_waiting_sec": queue_sec,
        "queue_wait_sec": queue_sec,
        "manifest_prep_sec": stages.get("manifest_preparation", {}).get("duration_sec", 0.0),
        "kaggle_submission_sec": stages.get("kaggle_submission", {}).get("duration_sec", 0.0),
        "flux_generation_sec": flux_sec,
        "artifact_download_sec": stages.get("artifact_download", {}).get("duration_sec", 0.0),
        "artifact_retrieval_sec": stages.get("artifact_download", {}).get("duration_sec", 0.0),
        "image_validation_sec": stages.get("image_validation", {}).get("duration_sec", 0.0),
        "validation_sec": stages.get("image_validation", {}).get("duration_sec", 0.0),
        "segmentation_sec": seg_sec,
        "deformation_sec": def_sec,
        "persistence_sec": stages.get("persistence", {}).get("duration_sec", 0.0),
        "total_elapsed_sec": total_elapsed,
        "estimated_remaining_sec": estimate_remaining_seconds(job),
        "completed_viewpoints": data.get("completed_viewpoints", 0),
        "stages": stages,
        "per_view_timing": data.get("per_view_timing", {}),
        "viewpoints": data.get("per_view_timing", {}),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }
