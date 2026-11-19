import urllib.request
import json

base_url = "http://127.0.0.1:8000"

print("Checking live backend...")
try:
    with urllib.request.urlopen(f"{base_url}/health", timeout=3) as resp:
        print("Health status:", resp.status, resp.read().decode())
    
    with urllib.request.urlopen(f"{base_url}/languages", timeout=3) as resp:
        langs = json.loads(resp.read().decode())
        print(f"Languages endpoint live! Returned {len(langs)} languages.")

    with urllib.request.urlopen(f"{base_url}/datasets/9/export", timeout=5) as resp:
        print("Export endpoint live! Status:", resp.status)
        print("Content-Type:", resp.headers.get("Content-Type"))
        print("Content-Disposition:", resp.headers.get("Content-Disposition"))
        content = resp.read()
        print(f"Downloaded ZIP size: {len(content) / 1024:.1f} KB")

except Exception as e:
    print("Live check note:", e)
