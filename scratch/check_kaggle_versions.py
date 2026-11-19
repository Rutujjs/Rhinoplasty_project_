import subprocess
import sys

for venv_path in [".venv", "backend/venv", "backend/venv_new"]:
    py_path = f"{venv_path}/Scripts/python.exe"
    try:
        res = subprocess.run(
            [py_path, "-m", "pip", "show", "kaggle"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in res.stdout.splitlines():
            if line.startswith("Name:") or line.startswith("Version:"):
                print(f"{venv_path}: {line}")
    except Exception as e:
        print(f"Error checking {venv_path}: {e}")
