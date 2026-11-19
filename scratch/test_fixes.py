import sys
import hashlib
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database.session import SessionLocal
from backend.app.models.dataset import Dataset
from backend.app.models.generated_image import GeneratedImage
from backend.app.models.generation_job import GenerationJob

client = TestClient(app)

print("--- 1. Testing Generation Endpoint with Dataset 8 ---")
# Dataset 8 should be tested without launching a new Kaggle job
gen_resp = client.post("/datasets/8/generate")
print(f"POST /datasets/8/generate status code: {gen_resp.status_code}")
print(f"POST /datasets/8/generate response: {gen_resp.json()}")
assert gen_resp.status_code in (202, 409), f"Unexpected status: {gen_resp.status_code}"

# Test GET generation-status
stat_resp = client.get("/datasets/8/generation-status")
print(f"GET /datasets/8/generation-status: {stat_resp.status_code}, {stat_resp.json()}")
assert stat_resp.status_code == 200

print("\n--- 2. Testing Batch Deformation on Dataset 8 ---")
batch_payload = {
    "deformation_type": "saddle",
    "strength": 0.45,
    "direction": None,
    "views": "all",
}
batch_resp = client.post("/datasets/8/deformation/batch", json=batch_payload)
print(f"POST /datasets/8/deformation/batch status: {batch_resp.status_code}")
data = batch_resp.json()
print(f"Succeeded count: {data.get('succeeded_count')}, Status: {data.get('status')}")
assert batch_resp.status_code == 200
assert data.get("succeeded_count") == 9

print("\n--- 3. Verifying Original and Deformed Image Endpoints ---")
db = SessionLocal()
images = db.query(GeneratedImage).filter(GeneratedImage.dataset_id == 8).all()
print(f"Dataset 8 images in DB: {len(images)}")
assert len(images) == 9

first_img = images[0]
orig_resp = client.get(f"/datasets/8/images/{first_img.id}")
print(f"GET /datasets/8/images/{first_img.id} status: {orig_resp.status_code}, content-type: {orig_resp.headers.get('content-type')}")
assert orig_resp.status_code == 200

def_resp = client.get(f"/datasets/8/deformation/image/{first_img.id}")
print(f"GET /datasets/8/deformation/image/{first_img.id} status: {def_resp.status_code}, length: {len(def_resp.content)}")
assert def_resp.status_code == 200

print("\n--- 4. Verifying Original P01 FLUX Images Immutability ---")
orig_p01_dir = Path("storage/generated/P01/original")
expected_frontal_prefix = "5c686563"
frontal_bytes = (orig_p01_dir / "frontal.png").read_bytes()
h = hashlib.sha256(frontal_bytes).hexdigest()
print(f"P01 frontal.png SHA-256: {h}")
assert h.startswith(expected_frontal_prefix), f"Original image mutated! {h}"

print("\n--- ALL CHECKS PASSED SUCCESSFULLY! ---")
db.close()

