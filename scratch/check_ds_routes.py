with open("backend/app/api/routes/datasets.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if "/images" in line or "image" in line.lower() or "def get_" in line:
            print(f"{idx}: {line.strip()[:100]}")
