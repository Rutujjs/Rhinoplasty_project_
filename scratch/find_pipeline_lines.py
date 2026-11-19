with open('nasal-deformation-pipeline-main/src/deformation.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'def deviated_nose_deformation' in line or 'def helicopter_deviated' in line:
        print(f"Line {i+1}: {line.strip()}")

