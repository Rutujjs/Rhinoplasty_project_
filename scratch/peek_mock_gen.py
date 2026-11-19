with open('generation/mock_generator.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if any(k in line.lower() for k in ['storage', 'generated', 'patient', 'dest', 'path']):
        print(f"Line {i+1}: {line.strip()[:100]}")
