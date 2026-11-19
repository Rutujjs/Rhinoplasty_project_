with open('backend/worker.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if any(k in line.lower() for k in ['storage', 'generated', 'image_path', 'path', 'p01']):
        print(f"Line {i+1}: {line.strip()[:100]}")
