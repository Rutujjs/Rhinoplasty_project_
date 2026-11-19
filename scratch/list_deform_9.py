import os

for root, dirs, files in os.walk('storage/deformations/9'):
    for f in files:
        print(os.path.join(root, f))

