from pathlib import Path

env_path = Path("backend/.env")
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if any(s in k.upper() for s in ["KEY", "SECRET", "PASSWORD", "TOKEN"]):
                print(f"{k} = [REDACTED length {len(v)}]")
            else:
                print(f"{k} = {v}")
else:
    print("backend/.env not found")
