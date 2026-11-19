import os
import shutil
import sqlite3
import hashlib
from pathlib import Path

BASE_DIR = Path(".").resolve()
STORAGE_DIR = BASE_DIR / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(str(STORAGE_DIR / "apac_face.db"))
cursor = conn.cursor()

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

# 1. Dataset 7 (Job 5)
print("=== Migrating Dataset 7 from Job 5 artifacts ===")
ds7_target_dir = DATASETS_DIR / "dataset_7" / "P01" / "original"
ds7_target_dir.mkdir(parents=True, exist_ok=True)

job5_root = STORAGE_DIR / "artifacts_tmp" / "job_5" / "rhinoplasty_dataset" / "patients" / "P01"
viewpoint_map_job5 = {
    "frontal": job5_root / "frontal" / "frontal_face.png",
    "basal": job5_root / "viewpoints" / "P01_basal.png",
    "helicopter": job5_root / "viewpoints" / "P01_helicopter.png",
    "left_20": job5_root / "viewpoints" / "P01_left_20.png",
    "left_40": job5_root / "viewpoints" / "P01_left_40.png",
    "left_60": job5_root / "viewpoints" / "P01_left_60.png",
    "right_20": job5_root / "viewpoints" / "P01_right_20.png",
    "right_40": job5_root / "viewpoints" / "P01_right_40.png",
    "right_60": job5_root / "viewpoints" / "P01_right_60.png",
}

for vk, src in viewpoint_map_job5.items():
    dest = ds7_target_dir / f"{vk}.png"
    if src.exists():
        shutil.copy2(src, dest)
        c_hash = sha256_of_file(dest)
        sz = dest.stat().st_size
        cursor.execute(
            "UPDATE generated_images SET image_path = ?, file_size_bytes = ?, checksum_sha256 = ? "
            "WHERE dataset_id = 7 AND viewpoint = ?",
            (str(dest), sz, c_hash, vk)
        )
        print(f"  DS7 {vk}: {dest.name} ({sz} bytes, {c_hash[:8]})")
    else:
        print(f"  DS7 {vk}: MISSING SRC {src}")

# 2. Dataset 4 (Job 2)
print("\n=== Migrating Dataset 4 from Job 2 artifacts ===")
ds4_target_dir = DATASETS_DIR / "dataset_4" / "P01" / "original"
ds4_target_dir.mkdir(parents=True, exist_ok=True)
job2_root = STORAGE_DIR / "artifacts_tmp" / "job_2" / "rhinoplasty_dataset" / "patients" / "P01"
viewpoint_map_job2 = {
    "frontal": job2_root / "frontal" / "frontal_face.png",
    "basal": job2_root / "viewpoints" / "P01_basal.png",
    "helicopter": job2_root / "viewpoints" / "P01_helicopter.png",
    "left_20": job2_root / "viewpoints" / "P01_left_20.png",
    "left_40": job2_root / "viewpoints" / "P01_left_40.png",
    "left_60": job2_root / "viewpoints" / "P01_left_60.png",
    "right_20": job2_root / "viewpoints" / "P01_right_20.png",
    "right_40": job2_root / "viewpoints" / "P01_right_40.png",
    "right_60": job2_root / "viewpoints" / "P01_right_60.png",
}
for vk, src in viewpoint_map_job2.items():
    dest = ds4_target_dir / f"{vk}.png"
    if src.exists():
        shutil.copy2(src, dest)
        c_hash = sha256_of_file(dest)
        sz = dest.stat().st_size
        cursor.execute(
            "UPDATE generated_images SET image_path = ?, file_size_bytes = ?, checksum_sha256 = ? "
            "WHERE dataset_id = 4 AND viewpoint = ?",
            (str(dest), sz, c_hash, vk)
        )
        print(f"  DS4 {vk}: {dest.name} ({sz} bytes, {c_hash[:8]})")

# 3. Dataset 6 (Job 4)
print("\n=== Migrating Dataset 6 from Job 4 artifacts ===")
ds6_target_dir = DATASETS_DIR / "dataset_6" / "P01" / "original"
ds6_target_dir.mkdir(parents=True, exist_ok=True)
job4_root = STORAGE_DIR / "artifacts_tmp" / "job_4" / "rhinoplasty_dataset" / "patients" / "P01"
viewpoint_map_job4 = {
    "frontal": job4_root / "frontal" / "frontal_face.png",
    "basal": job4_root / "viewpoints" / "P01_basal.png",
    "helicopter": job4_root / "viewpoints" / "P01_helicopter.png",
    "left_20": job4_root / "viewpoints" / "P01_left_20.png",
    "left_40": job4_root / "viewpoints" / "P01_left_40.png",
    "left_60": job4_root / "viewpoints" / "P01_left_60.png",
    "right_20": job4_root / "viewpoints" / "P01_right_20.png",
    "right_40": job4_root / "viewpoints" / "P01_right_40.png",
    "right_60": job4_root / "viewpoints" / "P01_right_60.png",
}
for vk, src in viewpoint_map_job4.items():
    dest = ds6_target_dir / f"{vk}.png"
    if src.exists():
        shutil.copy2(src, dest)
        c_hash = sha256_of_file(dest)
        sz = dest.stat().st_size
        cursor.execute(
            "UPDATE generated_images SET image_path = ?, file_size_bytes = ?, checksum_sha256 = ? "
            "WHERE dataset_id = 6 AND viewpoint = ?",
            (str(dest), sz, c_hash, vk)
        )
        print(f"  DS6 {vk}: {dest.name} ({sz} bytes, {c_hash[:8]})")

# 4. Dataset 9 (Current P01 generated images)
print("\n=== Migrating Dataset 9 from current generated images ===")
ds9_target_dir = DATASETS_DIR / "dataset_9" / "P01" / "original"
ds9_target_dir.mkdir(parents=True, exist_ok=True)
gen_p01 = STORAGE_DIR / "generated" / "P01" / "original"
for f in gen_p01.glob("*.png"):
    vk = f.stem
    dest = ds9_target_dir / f.name
    shutil.copy2(f, dest)
    c_hash = sha256_of_file(dest)
    sz = dest.stat().st_size
    cursor.execute(
        "UPDATE generated_images SET image_path = ?, file_size_bytes = ?, checksum_sha256 = ? "
        "WHERE dataset_id = 9 AND viewpoint = ?",
        (str(dest), sz, c_hash, vk)
    )
    print(f"  DS9 {vk}: {dest.name} ({sz} bytes, {c_hash[:8]})")

# 5. Dataset 8 (Korea, East Asian -> use distinct identity H02 from identities)
print("\n=== Migrating Dataset 8 with distinct identity H02 ===")
ds8_target_dir = DATASETS_DIR / "dataset_8" / "P01" / "original"
ds8_target_dir.mkdir(parents=True, exist_ok=True)
h02_dir = BASE_DIR / "nasal-deformation-pipeline-main" / "data" / "identities" / "H02" / "input"
if h02_dir.exists():
    for f in h02_dir.glob("*.png"):
        vk = f.stem
        dest = ds8_target_dir / f.name
        shutil.copy2(f, dest)
        c_hash = sha256_of_file(dest)
        sz = dest.stat().st_size
        cursor.execute(
            "UPDATE generated_images SET image_path = ?, file_size_bytes = ?, checksum_sha256 = ? "
            "WHERE dataset_id = 8 AND viewpoint = ?",
            (str(dest), sz, c_hash, vk)
        )
        print(f"  DS8 {vk}: {dest.name} ({sz} bytes, {c_hash[:8]})")

# 6. Dataset 3 (Identity H03)
print("\n=== Migrating Dataset 3 with distinct identity H03 ===")
ds3_target_dir = DATASETS_DIR / "dataset_3" / "P01" / "original"
ds3_target_dir.mkdir(parents=True, exist_ok=True)
h03_dir = BASE_DIR / "nasal-deformation-pipeline-main" / "data" / "identities" / "H03" / "input"
if h03_dir.exists():
    for f in h03_dir.glob("*.png"):
        vk = f.stem
        dest = ds3_target_dir / f.name
        shutil.copy2(f, dest)
        c_hash = sha256_of_file(dest)
        sz = dest.stat().st_size
        cursor.execute(
            "UPDATE generated_images SET image_path = ?, file_size_bytes = ?, checksum_sha256 = ? "
            "WHERE dataset_id = 3 AND viewpoint = ?",
            (str(dest), sz, c_hash, vk)
        )
        print(f"  DS3 {vk}: {dest.name} ({sz} bytes, {c_hash[:8]})")

conn.commit()
conn.close()
print("\nMigration complete!")
