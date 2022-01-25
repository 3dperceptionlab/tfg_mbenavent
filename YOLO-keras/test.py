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

import pandas as pd
import ast
actions_file = pd.read_csv("/workspace/tfg_mbenavent/EPIC-KITCHENS/processed-labels/actions_per_noun-full.csv", delimiter=';')
for index, action in actions_file.iterrows():
    print(action['noun_id'])
    x = ast.literal_eval(action['verbs'])
    print(x)
    print(type(x))
    # x = [n.strip() for n in action['verbs']]
    # print(x)
    break
    # self.actions[int(action['noun_id'])] = action['verbs']