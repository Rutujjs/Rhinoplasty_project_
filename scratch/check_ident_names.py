from pathlib import Path

for ident in ["H01", "H06", "H07"]:
    p = Path(f"nasal-deformation-pipeline-main/data/identities/{ident}/input")
    if p.exists():
        files = list(p.glob("*.png"))
        print(ident, [f.name for f in files])
