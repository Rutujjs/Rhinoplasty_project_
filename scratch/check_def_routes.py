with open("backend/app/api/routes/deformation.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if "deformed_image_path" in line or "deformations" in line or "def get_deformed" in line or "def deform" in line:
            print(f"{idx}: {line.strip()[:100]}")
