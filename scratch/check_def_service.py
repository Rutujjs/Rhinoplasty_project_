with open("backend/app/services/deformation_service.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if "output_dir" in line.lower() or "save" in line.lower() or "deformations" in line.lower() or "deformed_image_path" in line:
            print(f"{idx}: {line.strip()[:100]}")
