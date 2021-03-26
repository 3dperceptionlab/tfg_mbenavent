from PIL import Image, ImageDraw
import tarfile
import numpy as np

tar = tarfile.open('/datasets/egoDailyDatabase.tar')
imageraw = tar.extractfile('egoDailyDatabase/images/subject3/eating/eating1/frame10014.jpg')
image = Image.open(imageraw)
print(image.size)
left = 545
top = 714
right = 818 # width en este caso es xmax
bottom = 974 # height en este caso es ymax
draw = ImageDraw.Draw(image)
top = max(0, np.floor(top + 0.5).astype('int32'))
left = max(0, np.floor(left + 0.5).astype('int32'))
bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
draw.rectangle([left, top , right , bottom ])

left = 1091
top = 702
right = 1325
bottom = 936
draw = ImageDraw.Draw(image)
top = max(0, np.floor(top + 0.5).astype('int32'))
left = max(0, np.floor(left + 0.5).astype('int32'))
bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
draw.rectangle([left, top , right , bottom ])

image.save('myimage.jpg')