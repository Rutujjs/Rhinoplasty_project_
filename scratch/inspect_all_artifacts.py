import os
from pathlib import Path

print("=== CHECKING ARTIFACT_DOWNLOAD_DIR / STORAGE ===")
for root, dirs, files in os.walk("storage"):
    # show dirs under storage/artifacts_tmp or storage/datasets or storage/deformations
    if any(k in root for k in ["artifacts_tmp", "deformations", "generated"]):
        rel = os.path.relpath(root, "storage")
        if len(files) > 0 and not any(k in rel for k in ["bisenet"]):
            print(f"{rel}: {len(files)} files -> {files[:5]}")
