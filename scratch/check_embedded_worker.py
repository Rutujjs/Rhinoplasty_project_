with open("backend/app/core/config.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if "RUN_EMBEDDED_WORKER" in line or "embedded" in line.lower():
            print(f"{idx}: {line.strip()[:100]}")
