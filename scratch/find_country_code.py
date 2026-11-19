import os

for root, dirs, files in os.walk('.'):
    if any(x in root for x in ['.git', '.venv', 'node_modules', 'venv']):
        continue
    for f in files:
        if f.endswith(('.py', '.html', '.js', '.json')):
            p = os.path.join(root, f)
            with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
                c = fh.read()
                if 'Japan' in c or 'Korea' in c or 'population_group' in c or 'language' in c.lower():
                    print(f"Found in {p}")
