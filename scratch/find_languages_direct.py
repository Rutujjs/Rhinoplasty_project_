import os

for base in [r"c:\Users\poter\Downloads", r"c:\Users\poter\Desktop", r"c:\Users\poter\Documents", r"c:\Users\poter"]:
    try:
        for f in os.listdir(base):
            if 'language' in f.lower():
                print(os.path.join(base, f))
    except Exception:
        pass
