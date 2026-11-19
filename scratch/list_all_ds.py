import sqlite3

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()
print("All datasets:")
for row in c.execute("SELECT id, dataset_name, patient_id, status, morphology_type, num_images, generated_count FROM datasets"):
    print(row)
conn.close()
