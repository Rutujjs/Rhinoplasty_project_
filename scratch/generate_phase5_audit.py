import os
import hashlib
import json
import cv2
import numpy as np

def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

dataset_id = 9
patient_id = "P01"
morphology = "deviated"

orig_dir = f"storage/generated/{patient_id}/original"
deformed_dir = f"storage/deformations/{dataset_id}/{patient_id}/{morphology}"
mask_dir = f"storage/diagnostics/bisenet_patient_masks"
out_audit_dir = "storage/diagnostics/deviated_p01_audit"
os.makedirs(out_audit_dir, exist_ok=True)

viewpoints = [
    ("frontal", "FRONT_0"),
    ("left_20", "LEFT_20"),
    ("left_40", "LEFT_40"),
    ("left_60", "LEFT_60"),
    ("right_20", "RIGHT_20"),
    ("right_40", "RIGHT_40"),
    ("right_60", "RIGHT_60"),
    ("basal", "BASAL"),
    ("helicopter", "HELICOPTER"),
]

audit_records = []

for vp, tag in viewpoints:
    orig_path = os.path.join(orig_dir, f"{vp}.png")
    deform_path = os.path.join(deformed_dir, f"{vp}_{morphology}.png")
    mask_path = os.path.join(deformed_dir, f"{vp}_mask.png")
    
    if not os.path.exists(mask_path):
        mask_path = os.path.join(mask_dir, f"bisenet_mask_{vp}.png")

    orig_img = cv2.imread(orig_path)
    deform_img = cv2.imread(deform_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        mask_img = np.zeros(orig_img.shape[:2], dtype=np.uint8)

    h, w = orig_img.shape[:2]
    total_pixels = h * w
    
    orig_hash = sha256(orig_path)
    deform_hash = sha256(deform_path)
    hashes_differ = (orig_hash != deform_hash)

    diff = cv2.absdiff(orig_img, deform_img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    max_pixel_diff = int(np.max(diff_gray))
    mean_pixel_diff = float(np.mean(diff_gray))
    
    changed_mask = diff_gray > 2
    changed_pixels_count = int(np.count_nonzero(changed_mask))
    changed_pixels_pct = float((changed_pixels_count / total_pixels) * 100.0)

    # Inside vs outside mask
    mask_bool = mask_img > 0
    changed_inside_mask = int(np.count_nonzero(changed_mask & mask_bool))
    pct_inside_mask = float((changed_inside_mask / max(changed_pixels_count, 1)) * 100.0)
    
    mean_diff_inside_mask = float(np.mean(diff_gray[mask_bool])) if np.count_nonzero(mask_bool) > 0 else 0.0

    record = {
        "viewpoint": vp,
        "original_sha256": orig_hash,
        "deformed_sha256": deform_hash,
        "hashes_differ": hashes_differ,
        "image_size": f"{w}x{h}",
        "max_pixel_diff": max_pixel_diff,
        "mean_pixel_diff": round(mean_pixel_diff, 3),
        "mean_diff_inside_mask": round(mean_diff_inside_mask, 3),
        "changed_pixels_count": changed_pixels_count,
        "changed_pixels_pct": round(changed_pixels_pct, 2),
        "pct_changed_inside_mask": round(pct_inside_mask, 1),
    }
    audit_records.append(record)

    # Generate 4-panel diagnostic visual comparison:
    # [Original] | [BiSeNet Mask Overlay] | [Deformed (Deviated)] | [Amplified Diff Jet Map]
    overlay = orig_img.copy()
    overlay[mask_bool] = (overlay[mask_bool].astype(np.float32) * 0.6 + np.array([0, 255, 0], dtype=np.float32) * 0.4).astype(np.uint8)

    diff_amp = np.clip(diff_gray.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
    diff_colored = cv2.applyColorMap(diff_amp, cv2.COLORMAP_JET)

    # Add labels
    def add_label(img_copy, text):
        out = img_copy.copy()
        cv2.rectangle(out, (5, 5), (200, 30), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    p1 = add_label(orig_img, f"Original: {vp}")
    p2 = add_label(overlay, f"BiSeNet Mask ({np.count_nonzero(mask_bool)} px)")
    p3 = add_label(deform_img, f"Deviated (Str: 0.45)")
    p4 = add_label(diff_colored, f"Diff (Max: {max_pixel_diff})")

    quad_strip = np.hstack([p1, p2, p3, p4])
    strip_path = os.path.join(out_audit_dir, f"{vp}_comparison_strip.png")
    cv2.imwrite(strip_path, quad_strip)

summary_json_path = os.path.join(out_audit_dir, "deviated_audit_summary.json")
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(audit_records, f, indent=2)

print("Saved audit summary to", summary_json_path)
for r in audit_records:
    print(f"{r['viewpoint']:<12} | Diff: {r['max_pixel_diff']:<4} | MeanInMask: {r['mean_diff_inside_mask']:<5} | ChgPx: {r['changed_pixels_count']:<6} ({r['changed_pixels_pct']}%) | Differ: {r['hashes_differ']}")
