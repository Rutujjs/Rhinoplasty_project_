import subprocess

pythons = [
    r"C:\Users\poter\AppData\Local\Programs\Python\Python39\python.exe",
    r"C:\Users\poter\AppData\Local\Programs\Python\Python313\python.exe",
    r"C:\Program Files\PyManager\python.exe",
    r"C:\Users\poter\Downloads\python.exe",
]

for py in pythons:
    print("=" * 50)
    print("Testing:", py)
    try:
        res = subprocess.run([py, "-m", "pip", "show", "kaggle"], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if line.startswith("Name:") or line.startswith("Version:") or line.startswith("Location:"):
                print(" ", line)
        if not res.stdout:
            print("  pip show kaggle: not installed or empty")
    except Exception as e:
        print("  Error:", e)
