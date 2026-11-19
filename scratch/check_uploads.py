import os
import tempfile
import glob

p = os.path.join(tempfile.gettempdir(), '.kaggle', 'uploads')
print('uploads dir:', p)
print('uploads dir exists:', os.path.exists(p))
if os.path.exists(p):
    for f in glob.glob(os.path.join(p, '*')):
        print('file:', f)
