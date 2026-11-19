import os

for root, dirs, files in os.walk('backend'):
    for f in files:
        if 'db' in f.lower() or 'session' in f.lower():
            print(os.path.join(root, f))
