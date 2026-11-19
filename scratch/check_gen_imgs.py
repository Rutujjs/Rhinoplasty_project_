import sqlite3

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()
for row in c.execute("SELECT id, dataset_id, job_id, batch_id, image_path, viewpoint, created_at FROM generated_images WHERE dataset_id IN (7, 8, 9)"):
    print(row)
conn.close()
