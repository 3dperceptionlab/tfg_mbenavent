import os

path = '/datasets/mit_indoor/Images'

with open('Annotations/classes.txt') as f:
        class_names = f.readlines()
class_names = [c.strip() for c in class_names]


with open('Annotations/mit_indoor_adl.txt','w') as out:
    for idx, class_name in enumerate(class_names):
        out.write('\n'.join([os.path.join(path, class_name, f) + "," + str(idx) for f in os.listdir(os.path.join(path, class_name))]))
        out.write('\n')