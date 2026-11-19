import sqlite3

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()
c.execute("SELECT dataset_id, viewpoint, image_path, deformed_image_path, deformation_status FROM generated_images WHERE dataset_id IN (7, 8, 9) ORDER BY dataset_id, id")
for r in c.fetchall():
    print(r)
conn.close()
