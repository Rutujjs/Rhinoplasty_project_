import sqlite3
conn = sqlite3.connect('storage/apac_face.db')
c = conn.cursor()
c.execute('SELECT dataset_id, viewpoint, deformation_status, image_path, deformed_image_path FROM generated_images WHERE dataset_id IN (6, 7, 8) ORDER BY dataset_id, id')
for r in c.fetchall():
    print(r)
