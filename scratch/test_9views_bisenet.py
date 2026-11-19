import cv2
import numpy as np
from pathlib import Path
import onnxruntime as ort

model_path = Path("nasal-deformation-pipeline-main/models/bisenet/weights/resnet18.onnx")
p01_dir = Path("storage/generated/P01/original")

session = ort.InferenceSession(str(model_path))

views = [
    "frontal",
    "left_20",
    "left_40",
    "left_60",
    "right_20",
    "right_40",
    "right_60",
    "basal",
    "helicopter",
]

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

out_test_dir = Path("scratch/bisenet_9views_test")
out_test_dir.mkdir(parents=True, exist_ok=True)

print("=== TESTING BISENET ON ALL 9 VIEWPOINTS ===")
for v in views:
    img_path = p01_dir / f"{v}.png"
    img_bgr = cv2.imread(str(img_path))
    h, w = img_bgr.shape[:2]
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    norm_img = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor = np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    
    outputs = session.run(None, {"input": tensor})
    pred_labels_512 = np.argmax(outputs[0][0], axis=0).astype(np.uint8)
    
    # Class 10 is nose
    nose_mask_512 = (pred_labels_512 == 10).astype(np.uint8) * 255
    nose_px_512 = np.count_nonzero(nose_mask_512)
    
    classes_present = np.unique(pred_labels_512)
    
    # Resize back
    nose_mask = cv2.resize(nose_mask_512, (w, h), interpolation=cv2.INTER_NEAREST)
    nose_px = np.count_nonzero(nose_mask)
    
    if nose_px > 0:
        ys, xs = np.where(nose_mask > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        bbox = f"x=[{xs.min()}..{xs.max()}], y=[{ys.min()}..{ys.max()}]"
    else:
        cx, cy = 0.0, 0.0
        bbox = "EMPTY"
        
    print(f"{v:<12}: shape={w}x{h}, nose_px={nose_px} ({nose_px/(w*h)*100:.2f}%), center=({cx:.1f}, {cy:.1f}), bbox={bbox}")
    
    # Save colorized label map and overlay
    overlay = img_bgr.copy()
    red = np.zeros_like(img_bgr)
    red[:, :] = (0, 0, 255)
    overlay[nose_mask > 0] = (0.45 * img_bgr[nose_mask > 0].astype(np.float32) + 0.55 * red[nose_mask > 0].astype(np.float32)).astype(np.uint8)
    
    cv2.imwrite(str(out_test_dir / f"overlay_{v}.png"), overlay)
    cv2.imwrite(str(out_test_dir / f"mask_{v}.png"), nose_mask)
    cv2.imwrite(str(out_test_dir / f"labels_{v}.png"), cv2.resize(pred_labels_512, (w, h), interpolation=cv2.INTER_NEAREST))

