import os
import sys
sys.path.insert(0, ".")
import json
import zipfile
import hashlib
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage
from backend.app.database.session import SessionLocal

db = SessionLocal()

dataset = db.query(Dataset).filter(Dataset.id == 9).first()
print("Dataset:", dataset.id, dataset.dataset_name, dataset.patient_id, dataset.status)

images = db.query(GeneratedImage).filter(GeneratedImage.dataset_id == 9).all()
print("Images count:", len(images))
for img in images:
    print(f"  {img.viewpoint}: {img.image_path} (exists={os.path.exists(img.image_path)}) | def_status={getattr(img, 'deformation_status', None)}")

pid = dataset.patient_id or f"P{dataset.id:02d}"
morph = dataset.morphology_type or "saddle"
deform_dir = Path("storage/deformations") / str(dataset.id) / pid / morph
print("Deform dir:", deform_dir, "exists=", deform_dir.exists())
if deform_dir.exists():
    for f in deform_dir.iterdir():
        print("   deform file:", f.name)

diag_dir = Path("storage/diagnostics/deviated_p01_audit")
print("Diag dir exists:", diag_dir.exists())

db.close()
