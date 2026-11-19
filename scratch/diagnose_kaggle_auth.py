import os
import sys
import json
from pathlib import Path

# Add project root
sys.path.insert(0, ".")

print("=== DIAGNOSING KAGGLE AUTHENTICATION ===")

# 1. Environment variables
env_user = os.environ.get("KAGGLE_USERNAME")
env_key = os.environ.get("KAGGLE_KEY")
env_cfg = os.environ.get("KAGGLE_CONFIG_DIR")
print(f"os.environ KAGGLE_USERNAME: {env_user!r}")
print(f"os.environ KAGGLE_KEY: {'[PRESENT len=' + str(len(env_key)) + ']' if env_key else 'None'}")
print(f"os.environ KAGGLE_CONFIG_DIR: {env_cfg!r}")

# 2. backend/.env
backend_env_path = Path("backend/.env")
if backend_env_path.exists():
    print("backend/.env exists: YES")
    for line in backend_env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if any(line.startswith(prefix) for prefix in ["KAGGLE_", "DATABASE_URL"]):
            k = line.split("=")[0].strip()
            v = line.split("=")[1].strip() if "=" in line else ""
            if "KEY" in k.upper():
                print(f"  backend/.env {k}: [PRESENT len={len(v)}]")
            else:
                print(f"  backend/.env {k}: {v}")
else:
    print("backend/.env exists: NO")

# 3. Settings object
from backend.app.core.config import settings
print(f"settings.KAGGLE_USERNAME: {getattr(settings, 'KAGGLE_USERNAME', None)!r}")
print(f"settings.KAGGLE_CONFIG_DIR: {getattr(settings, 'KAGGLE_CONFIG_DIR', None)!r}")
print(f"settings.KAGGLE_KERNEL_SLUG: {getattr(settings, 'KAGGLE_KERNEL_SLUG', None)!r}")
print(f"settings.KAGGLE_MANIFEST_DATASET_SLUG: {getattr(settings, 'KAGGLE_MANIFEST_DATASET_SLUG', None)!r}")

# 4. Standard kaggle.json location
home_kaggle = Path.home() / ".kaggle" / "kaggle.json"
print(f"Checking {home_kaggle}...")
try:
    if home_kaggle.exists():
        print(f"  {home_kaggle} exists: YES")
        try:
            data = json.loads(home_kaggle.read_text(encoding="utf-8"))
            print(f"  username in kaggle.json: {data.get('username')!r}")
            k = data.get("key", "")
            print(f"  key in kaggle.json: {'[PRESENT len=' + str(len(k)) + ']' if k else 'MISSING'}")
        except Exception as e:
            print(f"  Error reading kaggle.json: {e}")
    else:
        print(f"  {home_kaggle} exists: NO")
except Exception as e:
    print(f"  Access error for {home_kaggle}: {e}")

# 5. Custom KAGGLE_CONFIG_DIR if set
if getattr(settings, "KAGGLE_CONFIG_DIR", None):
    custom_kaggle = Path(settings.KAGGLE_CONFIG_DIR) / "kaggle.json"
    if custom_kaggle != home_kaggle:
        try:
            if custom_kaggle.exists():
                print(f"  custom {custom_kaggle} exists: YES")
                data = json.loads(custom_kaggle.read_text(encoding="utf-8"))
                print(f"  username in custom kaggle.json: {data.get('username')!r}")
                k = data.get("key", "")
                print(f"  key in custom kaggle.json: {'[PRESENT len=' + str(len(k)) + ']' if k else 'MISSING'}")
            else:
                print(f"  custom {custom_kaggle} exists: NO")
        except Exception as e:
            print(f"  Access error for {custom_kaggle}: {e}")

# 6. Test KaggleApi initialization and authentication
print("\nTesting KaggleApi initialization...")
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    print("Calling api.authenticate()...")
    api.authenticate()
    print("api.authenticate() SUCCESS")
    print(f"Authenticated as config_values username: {api.config_values.get('username')!r}")
    print(f"Authenticated config_values key length: {len(api.config_values.get('key', ''))}")
    
    # Check dataset accessibility
    slug = settings.KAGGLE_MANIFEST_DATASET_SLUG
    if slug:
        print(f"\nChecking dataset status for {slug}...")
        try:
            status = api.dataset_status(slug)
            print(f"dataset_status result: {status}")
        except Exception as d_err:
            print(f"dataset_status FAILED: {type(d_err).__name__}: {d_err}")
            
    # Check kernel status
    kslug = settings.KAGGLE_KERNEL_SLUG
    if kslug:
        print(f"\nChecking kernel status for {kslug}...")
        try:
            kstatus = api.kernels_status(kslug)
            print(f"kernels_status result: {kstatus}")
        except Exception as k_err:
            print(f"kernels_status FAILED: {type(k_err).__name__}: {k_err}")
except Exception as err:
    print(f"KaggleApi authentication FAILED: {type(err).__name__}: {err}")

print("=== DIAGNOSIS COMPLETE ===")
