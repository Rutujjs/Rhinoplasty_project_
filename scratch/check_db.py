import sqlite3

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()

print("--- DATASETS ---")
c.execute("SELECT id, name, patient_id, region, ethnicity, morphology, status FROM datasets")
for row in c.fetchall():
    print(row)

print("\n--- GENERATION JOBS ---")
c.execute("SELECT id, dataset_id, status, error_message, artifact_count FROM generation_jobs")
for row in c.fetchall():
    print(row)

print("\n--- GENERATED IMAGES BY DATASET ---")
c.execute("SELECT dataset_id, job_id, viewpoint, image_path FROM generated_images ORDER BY dataset_id, id")
for row in c.fetchall():
    print(row)

conn.close()
