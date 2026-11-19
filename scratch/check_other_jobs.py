from pathlib import Path

for jid in [2, 4]:
    pj = Path(f"storage/artifacts_tmp/job_{jid}/rhinoplasty_dataset/patients/P01")
    print(f"Job {jid} files:")
    for f in pj.rglob("*.png"):
        print(" ", f.relative_to(pj), f.stat().st_size)
