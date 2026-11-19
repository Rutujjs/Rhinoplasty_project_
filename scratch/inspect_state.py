with open('frontend/index.html', 'r', encoding='utf-8') as f:
    c = f.read()
print('index.html size:', len(c))
print('gallery in index.html:', 'gallery' in c.lower())

with open('frontend/app.js', 'r', encoding='utf-8') as f:
    js = f.read()
print('app.js size:', len(js))
print('test_gallery in app.js:', 'test_gallery' in js)

