import urllib.request
import json

payload = {
    "deformation_type": "deviated",
    "strength": 0.45,
    "direction": "right"
}

req = urllib.request.Request(
    "http://127.0.0.1:8000/datasets/9/deformation/batch",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(req) as resp:
        print("Status:", resp.status)
        res = json.loads(resp.read().decode("utf-8"))
        print("Success! Overall status:", res.get("status"))
        print("Completed count:", sum(1 for v in res.get("viewpoints", []) if v.get("deformation_status") == "COMPLETED"))
        for v in res.get("viewpoints", []):
            print(f"  {v.get('viewpoint')}: {v.get('deformation_status')} - {v.get('processing_notes')}")
except urllib.error.HTTPError as e:
    print("HTTPError:", e.code, e.read().decode("utf-8"))
except Exception as e:
    print("Error:", e)

