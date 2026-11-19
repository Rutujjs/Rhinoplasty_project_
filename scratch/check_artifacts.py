import os
from pathlib import Path

artifacts_dir = Path("storage/artifacts_tmp")
if artifacts_dir.exists():
    for job_dir in sorted(artifacts_dir.glob("job_*")):
        print(f"Job: {job_dir.name}")
        patients_dir = job_dir / "rhinoplasty_dataset" / "patients"
        if patients_dir.exists():
            for p in patients_dir.glob("*"):
                print(f"  Patient: {p.name}")
                for sub in p.glob("*"):
                    if sub.is_dir():
                        files = list(sub.glob("*.png"))
                        print(f"    Subdir {sub.name}: {len(files)} pngs -> {[f.name for f in files[:3]]}")
                    elif sub.suffix == ".png":
                        print(f"    File: {sub.name}")
