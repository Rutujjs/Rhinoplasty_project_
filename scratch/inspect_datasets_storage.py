import sqlite3
import os

conn = sqlite3.connect("storage/apac_face.db")
c = conn.cursor()

print("=== DATASETS 7, 8, 9 ===")
for row in c.execute("SELECT id, dataset_name, patient_id, region, population_group, status, morphology_type FROM datasets WHERE id IN (7, 8, 9)"):
    print(row)

print("\n=== GENERATED IMAGES FOR DATASETS 7, 8, 9 ===")
for row in c.execute("SELECT id, dataset_id, viewpoint, viewpoint_name, image_path, validation_status, deformation_status FROM generated_images WHERE dataset_id IN (7, 8, 9) ORDER BY dataset_id, id"):
    print(row)

print("\n=== ALL DATASET IDS IN GENERATED IMAGES ===")
for row in c.execute("SELECT dataset_id, COUNT(*), MIN(image_path), MAX(image_path) FROM generated_images GROUP BY dataset_id"):
    print(row)

conn.close()
