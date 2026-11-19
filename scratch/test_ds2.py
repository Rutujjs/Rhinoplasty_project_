import sys
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

print("--- Testing GET /datasets/2/export (uncompleted, no zip on disk) ---")
resp_uncompleted = client.get("/datasets/2/export")
print("Status code:", resp_uncompleted.status_code)
assert resp_uncompleted.status_code == 404
print("Passed: uncompleted dataset with no pre-existing zip returns 404!")
