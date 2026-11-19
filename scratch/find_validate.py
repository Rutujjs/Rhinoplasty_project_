with open('backend/app/services/deformation_service.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if '_validate_deformation_result' in line:
        print(f"Line {i+1}: {line.strip()}")

