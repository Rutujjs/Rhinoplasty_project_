import os

for f in os.listdir("storage/exports"):
    print(f, os.path.getsize(os.path.join("storage/exports", f)))
