with open('diagnostic_suite.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines[:100]):
    print(f"{i+1}: {line.strip()[:100]}")
