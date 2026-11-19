import os

targets = ["test_gallery", "/gallery"]
found = []

for root_dir in ["backend", "frontend", "generation"]:
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith((".py", ".html", ".js", ".css", ".json")):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for idx, line in enumerate(f, 1):
                            for t in targets:
                                if t in line:
                                    found.append(f"{fpath}:{idx}: {line.strip()}")
                except Exception as e:
                    pass

for match in found[:30]:
    print(match)
