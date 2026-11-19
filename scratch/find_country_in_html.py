with open('frontend/index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if any(k in line.lower() for k in ['country', 'population_group', 'region', 'language']):
        print(f"Line {i+1}: {line.strip()[:100]}")
