import sys
sys.path.insert(0, ".")

import hashlib
import sqlite3
from pathlib import Path
from backend.app.database.session import SessionLocal
from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage
from backend.app.services.deformation_service import deformation_service

def _compute_sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

print("--- Testing Real Database Isolation for Datasets 7, 8, 9 ---")
db_path = Path("storage/apac_face.db")
conn = sqlite3.connect(str(db_path))
c = conn.cursor()

# Check datasets 7, 8, 9 have 9 viewpoints each
for ds_id in [7, 8, 9]:
    c.execute("SELECT COUNT(*) FROM generated_images WHERE dataset_id = ?", (ds_id,))
    cnt = c.fetchone()[0]
    print(f"Dataset {ds_id} count: {cnt}")
    assert cnt == 9

# Check path isolation
for ds_id in [7, 8, 9]:
    c.execute("SELECT image_path FROM generated_images WHERE dataset_id = ?", (ds_id,))
    paths = [r[0] for r in c.fetchall()]
    for p in paths:
        assert f"dataset_{ds_id}" in p
        assert Path(p).exists()
print("All paths exist and are isolated in dataset_{id} directories!")

# Check distinct hashes across all viewpoints
viewpoints = ["frontal", "left_20", "left_40", "left_60", "right_20", "right_40", "right_60", "basal", "helicopter"]
for vp in viewpoints:
    hashes = {}
    for ds_id in [7, 8, 9]:
        c.execute("SELECT image_path FROM generated_images WHERE dataset_id = ? AND viewpoint = ?", (ds_id, vp))
        row = c.fetchone()
        p = Path(row[0])
        hashes[ds_id] = _compute_sha256(p.read_bytes())

    assert hashes[7] != hashes[8], f"{vp}: DS7 == DS8"
    assert hashes[7] != hashes[9], f"{vp}: DS7 == DS9"
    assert hashes[8] != hashes[9], f"{vp}: DS8 == DS9"

print("All 9 viewpoints are 100% distinct between Datasets 7, 8, and 9!")

print("\n--- Testing Ensure Dataset Images Never Steals ---")
db = SessionLocal()
new_ds = Dataset(
    dataset_name="Test_Isolation_Isolated_DS_Temp",
    patient_id="P01",
    region="Japan",
    population_group="East Asian",
    age_min=20,
    age_max=30,
    sex_or_attribute="female",
    skin_tone="fair",
    pose="frontal",
    expression="neutral",
    num_images=9,
    resolution=512,
    status="CREATED",
    generated_count=0,
)
db.add(new_ds)
db.commit()

try:
    imgs = deformation_service._ensure_dataset_images(db, new_ds, patient_id="P01")
    print(f"Ensured images for brand new P01 dataset: {len(imgs)} (expected 0)")
    assert len(imgs) == 0
    assert new_ds.generated_count == 0
finally:
    db.delete(new_ds)
    db.commit()
    db.close()

conn.close()
print("\nALL ISOLATION VERIFICATIONS PASSED SUCCESSFULLY!")
