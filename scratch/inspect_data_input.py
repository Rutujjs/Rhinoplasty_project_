import cv2
from pathlib import Path

for p in Path('nasal-deformation-pipeline-main/data/input').glob('*'):
    img = cv2.imread(str(p))
    if img is not None:
        print(f'{p.name:<25}: shape={img.shape}')

