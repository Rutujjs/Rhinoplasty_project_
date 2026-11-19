import sqlite3
from pathlib import Path

conn = sqlite3.connect('storage/apac_face.db')
c = conn.cursor()

print('=== DATASETS ===')
for row in c.execute('SELECT id, dataset_name, patient_id, status, morphology_type FROM datasets'):
    print(row)

print('\n=== GENERATED IMAGES FOR DATASET 9 ===')
for row in c.execute('SELECT id, dataset_id, viewpoint, image_path, deformed_image_path, mask_image_path, deformation_status, validation_status FROM generated_images WHERE dataset_id=9'):
    print(row)

print('\n=== GENERATED IMAGES FOR DATASET 7 ===')
for row in c.execute('SELECT id, dataset_id, viewpoint, image_path, deformed_image_path, mask_image_path, deformation_status, validation_status FROM generated_images WHERE dataset_id=7'):
    print(row)

print('\n=== GENERATED IMAGES FOR DATASET 8 ===')
for row in c.execute('SELECT id, dataset_id, viewpoint, image_path, deformed_image_path, mask_image_path, deformation_status, validation_status FROM generated_images WHERE dataset_id=8'):
    print(row)

