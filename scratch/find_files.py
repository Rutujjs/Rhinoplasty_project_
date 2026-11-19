import os
from pathlib import Path

for p in Path('.').rglob('*'):
    parts = p.parts
    if any(x in parts for x in ['.git', '.venv', 'venv', 'venv_new', '__pycache__', 'node_modules']):
        continue
    if p.is_file():
        name = p.name.lower()
        if 'kaggle' in name or 'env' in name or 'secret' in name or 'credential' in name or name.endswith('.json'):
            print(p)
