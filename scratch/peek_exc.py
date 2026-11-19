with open('backend/app/core/exceptions.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'Export' in line or 'class ' in line:
        print(f"{i+1}: {line.strip()}")
