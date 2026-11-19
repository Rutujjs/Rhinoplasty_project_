with open('backend/tests/test_nasal_deformation_audit.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip().startswith('def test_'):
        print(f"Line {i+1}: {line.strip()}")

