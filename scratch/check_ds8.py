import sqlite3

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()
c.execute("SELECT id, dataset_id, job_id, image_path, created_at, prompt FROM generated_images WHERE dataset_id = 8")
for r in c.fetchall():
    print(r)
conn.close()
