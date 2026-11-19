import os

for root, dirs, files in os.walk('.'):
    # skip .git, node_modules, .venv, venv
    if any(x in root for x in ['.git', 'node_modules', '.venv', 'venv']):
        continue
    for f in files:
        if f.endswith(('.py', '.js', '.html', '.ts', '.tsx', '.json')):
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    c = fh.read()
                    if 'test_gallery' in c:
                        print(f'Found in {p}')
            except Exception:
                pass

