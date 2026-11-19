import cv2
import numpy as np
from pathlib import Path
import json

ref_dir = Path('deformation/reference_masks')
p01_dir = Path('storage/generated/P01/original')

views = ['frontal', 'left_20', 'left_40', 'left_60', 'right_20', 'right_40', 'right_60', 'basal', 'helicopter']
aliases = {
    'frontal': 'frontal_0',
    'left_20': 'left20',
    'left_40': 'left40',
    'left_60': 'left60',
    'right_20': 'right20',
    'right_40': 'right40',
    'right_60': 'right60',
    'basal': 'basal_view',
    'helicopter': 'helicopter_view'
}

print('=== 1. CHECKING BISENET CHECKPOINT AND CODE ===')
bisenet_dir = Path('nasal-deformation-pipeline-main/models/bisenet')
print(f'bisenet_dir exists: {bisenet_dir.exists()}')
if bisenet_dir.exists():
    b_contents = list(bisenet_dir.iterdir())
    print(f'bisenet_dir contents: {[p.name for p in b_contents]}')
else:
    print('bisenet_dir does not exist')

weights_path = bisenet_dir / 'weights' / 'resnet18.pt'
print(f'Expected weights path ({weights_path}) exists: {weights_path.exists()}')

inference_py = bisenet_dir / 'inference.py'
print(f'Expected inference.py ({inference_py}) exists: {inference_py.exists()}')

print('\n=== 2. REFERENCE MASKS INSPECTION ===')
mask_stats = {}
for v in views:
    ref_sub = ref_dir / aliases[v]
    bin_path = ref_sub / 'nose_binary.png'
    lbl_path = ref_sub / 'labels.png'
    p01_path = p01_dir / f'{v}.png'
    
    bin_exists = bin_path.exists()
    p01_exists = p01_path.exists()
    
    p01_shape = None
    if p01_exists:
        p01_img = cv2.imread(str(p01_path))
        if p01_img is not None:
            p01_shape = p01_img.shape
            
    if bin_exists:
        mask = cv2.imread(str(bin_path), cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        nonzero = int(np.count_nonzero(mask))
        ys, xs = np.where(mask > 0)
        ymin, ymax = int(ys.min()), int(ys.max())
        xmin, xmax = int(xs.min()), int(xs.max())
        box_w, box_h = xmax - xmin, ymax - ymin
        cx, cy = float((xmin + xmax) / 2), float((ymin + ymax) / 2)
        
        stat = {
            'view': v,
            'mask_file': str(bin_path),
            'mask_shape': [h, w],
            'p01_shape': list(p01_shape) if p01_shape else None,
            'active_pixels': nonzero,
            'active_pct': round(float(nonzero / (w * h) * 100), 2),
            'bbox': {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'w': box_w, 'h': box_h},
            'center': [round(cx, 1), round(cy, 1)],
            'relative_center': [round(cx / w, 3), round(cy / h, 3)]
        }
        mask_stats[v] = stat
        print(f"{v:<12}: mask={w}x{h}, p01={p01_shape}, active={nonzero}px ({stat['active_pct']}%), "
              f"box=[x={xmin}..{xmax} (w={box_w}), y={ymin}..{ymax} (h={box_h})], "
              f"rel_center=({stat['relative_center'][0]:.3f}, {stat['relative_center'][1]:.3f})")
    else:
        print(f"{v:<12}: MASK MISSING!")

with open('scratch/mask_audit_stats.json', 'w') as f:
    json.dump(mask_stats, f, indent=2)
print('\nSaved mask statistics to scratch/mask_audit_stats.json')

