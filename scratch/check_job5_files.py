from pathlib import Path

p5 = Path("storage/artifacts_tmp/job_5/rhinoplasty_dataset/patients/P01")
print("Job 5 files:")
for f in p5.rglob("*.png"):
    print(" ", f.relative_to(p5), f.stat().st_size)
