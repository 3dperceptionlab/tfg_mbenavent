# import cv2
# from random import randint

# src = cv2.imread("sample1.jpg",cv2.IMREAD_COLOR)

# top = 0  # shape[0] = rows
# bottom = 40
# left = 0  # shape[1] = cols
# right = 0
# borderType = cv2.BORDER_CONSTANT
# value = [0, 0, 0]
# dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
# cv2.imwrite("border.png", dst)

# import pandas as pd
# import ast
# actions_file = pd.read_csv("/workspace/tfg_mbenavent/EPIC-KITCHENS/processed-labels/actions_per_noun-full.csv", delimiter=';')
# for index, action in actions_file.iterrows():
#     print(action['noun_id'])
#     x = ast.literal_eval(action['verbs'])
#     print(x)
#     print(type(x))
#     # x = [n.strip() for n in action['verbs']]
#     # print(x)
#     break
#     # self.actions[int(action['noun_id'])] = action['verbs']


#!/usr/bin/env python
import cv2
 
img = cv2.imread('sample4.jpg')
 
# (1) create a copy of the original:
overlay = img.copy()
# (2) draw shapes:
cv2.circle(overlay, (500, 500), 600, (255, 0, 0), -1)
# (3) blend with the original:
opacity = 0.2
cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
 
cv2.imwrite('pipeline_result.png', img)