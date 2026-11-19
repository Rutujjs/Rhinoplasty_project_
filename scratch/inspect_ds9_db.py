import sqlite3
import json

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()

print("--- DATASET 9 ---")
for row in c.execute("SELECT * FROM datasets WHERE id = 9"):
    print(row)

print("--- GENERATED IMAGES FOR DATASET 9 ---")
for row in c.execute("SELECT id, dataset_id, viewpoint, viewpoint_name, image_path, validation_status, deformation_status FROM generated_images WHERE dataset_id = 9"):
    print(row)

print("--- DEFORMATION AUDIT / TABLES ---")
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", c.fetchall())

conn.close()
