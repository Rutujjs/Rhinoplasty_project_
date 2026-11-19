import hashlib
from pathlib import Path

def hash_file(p):
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:12]

p_gen = Path("storage/generated/P01/original/frontal.png")
p_job5 = Path("storage/artifacts_tmp/job_5/rhinoplasty_dataset/patients/P01/frontal/frontal_face.png")
p_job4 = Path("storage/artifacts_tmp/job_4/rhinoplasty_dataset/patients/P01/frontal/frontal_face.png")
p_job2 = Path("storage/artifacts_tmp/job_2/rhinoplasty_dataset/patients/P01/frontal/frontal_face.png")

print("Generated P01 frontal:", hash_file(p_gen))
print("Job 5 (DS 7) frontal: ", hash_file(p_job5))
print("Job 4 (DS 6) frontal: ", hash_file(p_job4))
print("Job 2 (DS 4) frontal: ", hash_file(p_job2))
