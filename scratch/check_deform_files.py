from pathlib import Path

p = Path("storage/deformations")
if p.exists():
    for f in p.rglob("*.png"):
        print(f.relative_to(p))
else:
    print("No storage/deformations dir")
