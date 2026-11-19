import subprocess
import sys
import shutil

print("Current sys.executable:", sys.executable)

# Check 'where python'
try:
    res = subprocess.run(["where", "python"], capture_output=True, text=True)
    print("where python:\n", res.stdout)
except Exception as e:
    print("where python failed:", e)

# Check 'where uvicorn'
try:
    res = subprocess.run(["where", "uvicorn"], capture_output=True, text=True)
    print("where uvicorn:\n", res.stdout)
except Exception as e:
    print("where uvicorn failed:", e)

# Check 'where kaggle'
try:
    res = subprocess.run(["where", "kaggle"], capture_output=True, text=True)
    print("where kaggle:\n", res.stdout)
except Exception as e:
    print("where kaggle failed:", e)
