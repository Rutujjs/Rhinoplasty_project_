import os

for root, dirs, files in os.walk('storage'):
    for f in files:
        if 'mask' in f.lower():
            print(os.path.join(root, f))

