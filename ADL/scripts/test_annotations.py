import cv2

image = cv2.imread('/datasets/ADL/rgb_frames/P_02/002130.jpg')

#/datasets/ADL/rgb_frames/P_02/002130.jpg ,24 297,203,383,258,10 142,171,165,194,23 138,0,213,107,19 240,122,292,216,9 195,3,262,99,19 218,218,248,243,37 494,103,530,168,24 250,0,424,103,18

# width = int(image.shape[1] * 0.5)
# height = int(image.shape[0] * 0.5)
# dim = (width, height)

# image = cv2.resize(image, dim)

xmin, ymin, xmax, ymax = (297,203,383,258)

color = (0,0,255)
cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3, cv2.LINE_AA)

cv2.imwrite('res.jpg', image)
