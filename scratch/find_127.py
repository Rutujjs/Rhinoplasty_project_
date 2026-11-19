with open('deformation/algorithms.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if '> 127' in line:
        print(f"Line {i+1}: {line.strip()}")

