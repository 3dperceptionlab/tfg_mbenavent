from PIL import Image, ImageDraw
import tarfile
import numpy as np

with open('train.txt') as f:
    lines = f.readlines()
    line = lines[0].split()
    tar = tarfile.open(line[0])
    imageraw = tar.extractfile(line[1])
    image = Image.open(imageraw)
    print(image.size)
    boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[2:]])
    left = 1260
    top = 76
    right = 1446
    bottom = 538
    draw = ImageDraw.Draw(image)
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    draw.rectangle([left, top , right , bottom ])
    image.save('myimage.jpg')

    # for box in boxes:
    #     left = box[0]
    #     top = box[1]
    #     right = box[2]
    #     bottom = box[3]
    #     top = max(0, np.floor(top + 0.5).astype('int32'))
    #     left = max(0, np.floor(left + 0.5).astype('int32'))
    #     bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    #     right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    #     draw = ImageDraw.Draw(image)
    #     for i in range(thickness):
    #         draw.rectangle(
    #             [left + i, top + i, right - i, bottom - i])
    #         break
    #     image.save('test.jpg')