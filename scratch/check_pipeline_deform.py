with open('nasal-deformation-pipeline-main/src/deformation.py', 'r', encoding='utf-8') as f:
    c = f.read()
if 'deviated_nose_deformation' in c:
    print('Found deviated_nose_deformation in nasal-deformation-pipeline-main/src/deformation.py')
else:
    print('Not found')

