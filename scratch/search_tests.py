import os

for root, _, files in os.walk("backend/tests"):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                content = fp.read()
                if "generate_multiview" in content or "mock_generator" in content or "draw_mock_image" in content:
                    print(f"Match in {p}")
