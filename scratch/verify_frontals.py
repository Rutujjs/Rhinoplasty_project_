import sqlite3
import hashlib
from pathlib import Path

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()
c.execute("SELECT dataset_id, viewpoint, image_path FROM generated_images WHERE viewpoint = 'frontal' ORDER BY dataset_id")
rows = c.fetchall()
conn.close()

print(f"{'DS ID':<6} {'Viewpoint':<10} {'File Exists?':<14} {'SHA256 (first 10)':<18} {'Image Path'}")
print("-" * 80)
for ds_id, vp, p_str in rows:
    p = Path(p_str)
    exists = p.exists()
    sha = hashlib.sha256(p.read_bytes()).hexdigest()[:10] if exists else "N/A"
    print(f"{ds_id:<6} {vp:<10} {str(exists):<14} {sha:<18} {p_str}")
