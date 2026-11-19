import sys
sys.path.insert(0, ".")
import zipfile
from backend.app.database.session import SessionLocal
from backend.app.models.dataset import Dataset
from backend.app.services.export_service import build_export_zip

db = SessionLocal()
dataset = db.query(Dataset).filter(Dataset.id == 9).first()
assert dataset is not None, "Dataset 9 not found!"

zip_path = build_export_zip(db, dataset)
print(f"Generated ZIP: {zip_path} (Size: {zip_path.stat().st_size / 1024:.1f} KB)")

zf = zipfile.ZipFile(zip_path)
print("\nArchive File List:")
for name in sorted(zf.namelist()):
    info = zf.getinfo(name)
    print(f"  {name:<50} ({info.file_size / 1024:.1f} KB)")

# Verify key files exist
expected_files = [
    "dataset_metadata.json",
    "README.txt",
    "manifest.txt",
    "deformation_audit.json",
    "patients/P01/metadata.json",
    "patients/P01/original/frontal.png",
    "patients/P01/original/basal.png",
    "patients/P01/original/helicopter.png",
    "patients/P01/deformed/frontal_deviated.png",
    "patients/P01/masks/frontal_mask.png",
]
for exp in expected_files:
    assert exp in zf.namelist(), f"Missing required file in ZIP: {exp}"

print("\nALL EXPECTED FILES PRESENT AND VERIFIED!")
db.close()
