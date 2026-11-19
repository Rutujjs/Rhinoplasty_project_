import pathlib

for base in ['backend/venv', 'backend/venv_new']:
    for p in pathlib.Path(base).glob('**/*.py'):
        try:
            txt = p.read_text(encoding='utf-8', errors='ignore')
            if 'please consider upgrading' in txt.lower():
                print(p)
        except Exception:
            pass
