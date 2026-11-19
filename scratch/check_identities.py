from pathlib import Path

p = Path("nasal-deformation-pipeline-main/data/identities")
if p.exists():
    for ident in p.iterdir():
        if ident.is_dir():
            files = list((ident / "input").glob("*.png")) if (ident / "input").exists() else []
            print(f"Identity: {ident.name}, {len(files)} files in input")
else:
    print("Identities dir does not exist")
