import urllib.request
import json

req = urllib.request.Request("http://127.0.0.1:8000/datasets/9/deformation/status")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    print("Dataset 9 status:", data.get("status"))
    print("Active type:", data.get("active_deformation_type"))
    print("Viewpoints count:", len(data.get("viewpoints", [])))
    for v in data.get("viewpoints", []):
        print(f"  {v['viewpoint']}: status={v['deformation_status']}, url={v['deformed_image_url']}")
