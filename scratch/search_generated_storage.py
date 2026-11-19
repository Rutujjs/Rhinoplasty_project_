import os
import re

targets = [r"storage[\\/]generated", r"generated_dir", r"OUTPUT_DIR", r"patient_orig_dir"]

for root_dir in ["backend", "generation"]:
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".py"):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for idx, line in enumerate(f, 1):
                            if any(re.search(t, line) for t in targets):
                                print(f"{fpath}:{idx}: {line.strip()[:100]}")
                except Exception:
                    pass
