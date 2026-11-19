import sqlite3

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()
print("=== GENERATION JOBS ===")
for row in c.execute("SELECT id, dataset_id, status, provider, requested_count, generated_count, batch_id, created_at FROM generation_jobs"):
    print(row)
conn.close()
