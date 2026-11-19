with open("frontend/index.html", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines, 1):
    if any(k in l for k in ["/gallery", "gallery", "loadImages", "viewpoint", "test_gallery"]):
        print(f"{i}: {l.strip()[:100]}")
