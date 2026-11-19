import sys
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

print("--- Testing GET /datasets/9/export ---")
resp = client.get("/datasets/9/export")
print("Status code:", resp.status_code)
print("Content-Type:", resp.headers.get("content-type"))
print("Content-Disposition:", resp.headers.get("content-disposition"))
print("Content length (KB):", len(resp.content) / 1024)
assert resp.status_code == 200
assert resp.headers.get("content-type") == "application/zip"
assert "dataset_9_export.zip" in resp.headers.get("content-disposition")

print("\n--- Testing GET /datasets/999999/export (nonexistent) ---")
resp404 = client.get("/datasets/999999/export")
print("Status code:", resp404.status_code)
assert resp404.status_code == 404

print("\n--- Testing GET /datasets/1/export (uncompleted, 0 images) ---")
resp_uncompleted = client.get("/datasets/1/export")
print("Status code:", resp_uncompleted.status_code)
assert resp_uncompleted.status_code == 404

print("\nALL API EXPORT TESTS PASSED!")
