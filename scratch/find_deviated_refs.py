import os

for root, dirs, files in os.walk('.'):
    if any(x in root for x in ['.git', 'node_modules', '.venv', 'venv']):
        continue
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            with open(p, 'r', encoding='utf-8') as fh:
                c = fh.read()
                if 'deviated_nose_deformation' in c:
                    print(f"Found in {p}")

