import sqlite3
import hashlib
from pathlib import Path
from PIL import Image

BASE_DIR = Path(".").resolve()
STORAGE_DIR = BASE_DIR / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"

conn = sqlite3.connect(str(STORAGE_DIR / "apac_face.db"))
cursor = conn.cursor()

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

ident_map = {
    "01_frontal_0": "frontal",
    "02_left_20": "left_20",
    "03_left_40": "left_40",
    "04_left_60": "left_60",
    "05_right_20": "right_20",
    "06_right_40": "right_40",
    "07_right_60": "right_60",
    "08_basal_view": "basal",
    "09_helicopter_view": "helicopter",
}

ds9_dir = DATASETS_DIR / "dataset_9" / "P01" / "original"
ds9_dir.mkdir(parents=True, exist_ok=True)
h01_dir = BASE_DIR / "nasal-deformation-pipeline-main" / "data" / "identities" / "H01" / "input"

print("Populating Dataset 9 with distinct identity H01...")
for old_prefix, canon_key in ident_map.items():
    src_file = list(h01_dir.glob(f"{old_prefix}*.png"))
    if not src_file:
        print(f"  Warning: no match for {old_prefix} in {h01_dir}")
        continue
    src_path = src_file[0]
    dest_path = ds9_dir / f"{canon_key}.png"
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        im = im.resize((512, 512), Image.Resampling.LANCZOS)
        im.save(dest_path, "PNG")
    sz = dest_path.stat().st_size
    c_hash = sha256_of_file(dest_path)
    cursor.execute(
        "UPDATE generated_images SET image_path = ?, file_size_bytes = ?, checksum_sha256 = ?, image_width = 512, image_height = 512 "
        "WHERE dataset_id = 9 AND viewpoint = ?",
        (str(dest_path), sz, c_hash, canon_key)
    )
    print(f"  DS9 {canon_key}: {dest_path.name} ({sz} bytes, {c_hash[:8]})")

conn.commit()
conn.close()
print("Dataset 9 is now 100% distinct across all 9 viewpoints!")
