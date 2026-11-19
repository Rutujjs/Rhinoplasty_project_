import sqlite3
import hashlib
from pathlib import Path

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()

for vp in ["frontal", "left_20", "left_40", "left_60", "right_20", "right_40", "right_60", "basal", "helicopter"]:
    c.execute("SELECT dataset_id, image_path FROM generated_images WHERE viewpoint = ? AND dataset_id IN (7, 8, 9) ORDER BY dataset_id", (vp,))
    rows = c.fetchall()
    hashes = {}
    for ds_id, p_str in rows:
        p = Path(p_str)
        hashes[ds_id] = hashlib.sha256(p.read_bytes()).hexdigest()[:8] if p.exists() else "MISSING"
    print(f"Viewpoint {vp:<12}: DS7={hashes.get(7)} | DS8={hashes.get(8)} | DS9={hashes.get(9)}")

conn.close()
