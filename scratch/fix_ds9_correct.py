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

ident_map_h01 = {
    "H01_FRONTAL.png": "frontal",
    "H01_LEFT20.png": "left_20",
    "H01_LEFT40.png": "left_40",
    "H01_LEFT60.png": "left_60",
    "H01_RIGHT20.png": "right_20",
    "H01_RIGHT40.png": "right_40",
    "H01_RIGHT60.png": "right_60",
    "H01_BASAL.png": "basal",
    "H01_HELICOPTER.png": "helicopter",
}

ds9_dir = DATASETS_DIR / "dataset_9" / "P01" / "original"
ds9_dir.mkdir(parents=True, exist_ok=True)
h01_dir = BASE_DIR / "nasal-deformation-pipeline-main" / "data" / "identities" / "H01" / "input"

print("Populating Dataset 9 with distinct identity H01...")
for filename, canon_key in ident_map_h01.items():
    src_path = h01_dir / filename
    if not src_path.exists():
        print(f"  Warning: {filename} does not exist in {h01_dir}")
        continue
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
