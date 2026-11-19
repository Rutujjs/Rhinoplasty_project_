import os

for root, _, files in os.walk("backend/tests"):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                for idx, line in enumerate(fp, 1):
                    if 'client.get("/images' in line or 'client.get(f"/images' in line or 'client.get(f"/datasets' in line:
                        print(f"{p}:{idx}: {line.strip()[:100]}")
