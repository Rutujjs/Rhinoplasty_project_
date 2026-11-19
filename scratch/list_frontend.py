import os

for f in os.listdir('frontend'):
    print(f, os.path.getsize(os.path.join('frontend', f)))

