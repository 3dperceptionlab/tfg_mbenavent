from PIL import Image, ImageDraw
import tarfile
import numpy as np


image = Image.open('/datasets/egoDailyDatabase/images/subject1/eating/eating1/frame1009.jpg')
#619,487,952,758,0,0.993899 934,466,1253,775,0,0.979289
left = 619
top = 487
right = 952
bottom = 758
draw = ImageDraw.Draw(image)
top = max(0, np.floor(top + 0.5).astype('int32'))
left = max(0, np.floor(left + 0.5).astype('int32'))
bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
draw.rectangle([left, top , right , bottom ], outline='red', width=2)

left = 934
top = 466
right = 1253
bottom = 775
draw = ImageDraw.Draw(image)
top = max(0, np.floor(top + 0.5).astype('int32'))
left = max(0, np.floor(left + 0.5).astype('int32'))
bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
draw.rectangle([left, top , right , bottom ], outline='blue', width=2)
#/datasets/egoDailyDatabase/images/subject1/eating/eating1/frame1009.jpg 612,477,343,313,0 955,480,277,280,0

left = 612
top = 477
right = left+343
bottom = top+313
draw = ImageDraw.Draw(image)
top = max(0, np.floor(top + 0.5).astype('int32'))
left = max(0, np.floor(left + 0.5).astype('int32'))
bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
draw.rectangle([left, top , right , bottom ], outline='green', width=2)

left = 955
top = 480
right = left+277
bottom = top+280
draw = ImageDraw.Draw(image)
top = max(0, np.floor(top + 0.5).astype('int32'))
left = max(0, np.floor(left + 0.5).astype('int32'))
bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
draw.rectangle([left, top , right , bottom ], outline='yellow', width=2)

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