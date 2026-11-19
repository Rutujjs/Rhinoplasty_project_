with open('backend/app/api/routes/deformation.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    if '@router' in line or 'def ' in line:
        print(line.strip())

