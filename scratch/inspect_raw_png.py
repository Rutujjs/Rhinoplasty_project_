import cv2
import numpy as np

for path in [
    'nasal-deformation-pipeline-main/data/diagnostics/bisenet_test/resnet18/1_raw.png',
    'nasal-deformation-pipeline-main/data/diagnostics/left60_mask_test/resnet18/left60_raw.png'
]:
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is not None:
        print(f"{path}: shape={raw.shape}, dtype={raw.dtype}, unique={np.unique(raw)}")
    else:
        print(f"{path}: NOT FOUND")

