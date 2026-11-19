import os

search_paths = [
    r"c:\Users\poter\Downloads",
    r"c:\Users\poter\Desktop",
    r"c:\Users\poter\Documents",
    r"c:\Users\poter\Downloads\synthetic",
]

for base in search_paths:
    if os.path.exists(base):
        for root, dirs, files in os.walk(base):
            if any(x in root for x in ['.git', '.venv', 'node_modules', 'venv', 'AppData']):
                continue
            for f in files:
                if 'language' in f.lower() or f.lower().endswith('.pdf'):
                    print(os.path.join(root, f))
