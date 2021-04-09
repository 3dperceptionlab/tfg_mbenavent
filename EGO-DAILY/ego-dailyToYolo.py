import csv
import sys
from collections import namedtuple
import ast
import math

if (len(sys.argv) != 3):
    print('[ERROR] Expected filename with classes.')
    print('Usage: python ego-dailyToYolo.py [train_object_labels] [test_object_labels]')
    exit()

classes = []
print('-------- PROCESSING CLASSES --------')

with open('ego-daily_classes.txt','w') as classes_file:
    classes_file.write('right_hand\n')
    classes_file.write('left_hand\n')
    print('There are 2 classes.')
    

print('-------- PROCESSING LABELS --------')
def process_labels (filename, outputname):
    with open(filename) as file:
        with open(outputname, 'w') as output:
            lines = file.readlines()
            line_count = 0
            location = ''
            remaining_objects = -2
            bounding_boxes = []
            for row in lines:
                if line_count == 0:
                    imgs = int(row.rstrip('\n'))
                    print(f'There are {imgs} images')
                    line_count += 1
                else:
                    line_count += 1
                    if not location:
                        location = row.rstrip('\n')
                    elif remaining_objects == -2: # Image size
                        remaining_objects = -1
                    elif remaining_objects == -1: # Number of objects
                        remaining_objects = int(row)
                    elif remaining_objects > 0:
                        row_split = row.split()
                        xmin = math.trunc(float(row_split[0]))
                        ymin = math.trunc(float(row_split[1]))
                        xmax = math.trunc(float(row_split[2]))
                        ymax = math.trunc(float(row_split[3]))
                        object_class = int(row_split[4])
                        bounding_boxes.append((xmin, ymin, xmax, ymax, object_class))
                        remaining_objects -= 1
                        if remaining_objects == 0:
                            if 'run' not in location and 'bike' not in location: # Save only interior locations (eating, kitchens, office)
                                output.write('/datasets/' + location + ' ')
                                for bb in bounding_boxes:
                                    output.write(str(bb[0]) + ',' + str(bb[1]) + ',' + str(bb[2]) + ',' + str(bb[3]) + ',' + str(bb[4] - 1) + ' ')
                                output.write('\n')
                            location = ''
                            remaining_objects = -2
                            bounding_boxes = []
                        


process_labels(sys.argv[1], 'train.txt')
process_labels(sys.argv[2], 'test.txt')