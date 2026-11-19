import sqlite3

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()
c.execute("SELECT id, dataset_name, patient_id, region, population_group, morphology_type, status, generated_count FROM datasets")
for r in c.fetchall():
    print(r)
conn.close()
