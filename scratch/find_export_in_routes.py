with open('backend/app/api/routes/datasets.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'export' in line.lower():
        print(f"Line {i+1}: {line.strip()}")
