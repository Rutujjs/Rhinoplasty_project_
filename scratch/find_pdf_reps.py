import os

for root, dirs, files in os.walk('.'):
    if any(x in root for x in ['.git', '.venv', 'node_modules', 'venv']):
        continue
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                c = fh.read()
                if 'pdf' in c.lower() or 'report' in c.lower() or 'country' in c.lower() or 'languages' in c.lower():
                    print(f"Found in {p}")
