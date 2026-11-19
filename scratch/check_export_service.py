with open("backend/app/services/export_service.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if "path" in line.lower() or "image" in line.lower():
            print(f"{idx}: {line.strip()[:100]}")
