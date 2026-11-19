import sqlite3

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()

c.execute("PRAGMA table_info(datasets)")
cols = [r[1] for r in c.fetchall()]
print("Dataset columns:", cols)

c.execute(f"SELECT {', '.join(cols[:8])} FROM datasets")
for row in c.fetchall():
    print(row)

print("\n--- GENERATION JOBS ---")
c.execute("SELECT id, dataset_id, status FROM generation_jobs")
for row in c.fetchall():
    print(row)

print("\n--- GENERATED IMAGES (sample) ---")
c.execute("SELECT dataset_id, job_id, viewpoint, image_path FROM generated_images GROUP BY dataset_id, viewpoint")
for row in c.fetchall():
    print(row)

conn.close()
