import urllib.request
import json

try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as resp:
        print("Backend is RUNNING! Status:", resp.status, resp.read().decode())
except Exception as e:
    print("Backend not reachable or error:", e)

