
from email.mime import image

images = []
with open('processed-labels/train_full.txt') as f:
    lines = f.readlines()
    for line in lines:
        line_divided = line.split()
        count = len(line_divided[1:])
        if count >= 5:
            images.append(line_divided[0])


with open('processed-labels/images_most_labels.txt','w') as f:
    for img in images:
        f.write(img + "\n")
