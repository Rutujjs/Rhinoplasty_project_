import sys
sys.path.insert(0, ".")

from backend.app.database.session import SessionLocal
from backend.app.models.dataset import Dataset
from backend.app.schemas.deformation import BatchDeformationRequest
from backend.app.services.deformation_service import deformation_service

db = SessionLocal()

tasks = [
    (7, "broad", 0.5, None),
    (8, "saddle", 0.5, None),
    (9, "deviated", 0.5, "left"),
]

for ds_id, morph, strength, direction in tasks:
    ds = db.query(Dataset).filter(Dataset.id == ds_id).first()
    print(f"\n--- Processing batch deformation for Dataset {ds_id} ({morph}) ---")
    req = BatchDeformationRequest(
        patient_id="P01",
        deformation_type=morph,
        strength=strength,
        direction=direction,
        force_recompute=True,
    )
    result = deformation_service.process_all_viewpoints(db, ds_id, req)
    print(f"Dataset {ds_id} result: status={result.status}, succeeded={result.succeeded_count}, failed={result.failed_count}")

db.close()
print("\nBatch deformations completed!")
