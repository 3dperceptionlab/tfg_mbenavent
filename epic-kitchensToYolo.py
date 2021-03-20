import csv
import csv
import sys
from collections import namedtuple
import ast

if (len(sys.argv) != 3):
    print('[ERROR] Expected filename with classes.')
    print('Usage: python epic-kitchensToYolo.py [object_labels] [noun_classes]')
    exit()

classes = []
with open(sys.argv[1]) as csv_file:
    with open('epic-kitchens_classes.txt','w') as classes_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif line_count == 1:
                print(f'Class "{row[1]}" is not used in YOLO.')
                line_count += 1
            else:
                classes_file.write(str(row[1]) + '\n')
                classes.append(row[1])
                line_count += 1
        print(f'Processed {line_count} lines.')


labels_dict = {}
LabelData = namedtuple("LabelData", "object_class bounding_box")

with open(sys.argv[2]) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            location = '/epic-kitchens/' + row[2] + '/object_detection_images/' + row[4] + '.jpg'
            bounding_boxes = x = ast.literal_eval(row[5])
            object_class = row[0] # we use id in order to avoid errors
            partial_labels = []
            for bb in bounding_boxes:
                label = LabelData(object_class=object_class, bounding_box=bb)
                partial_labels.append(label)
            if len(bounding_boxes) == 0:
                continue
            if location not in labels_dict:
                # Add
                labels_dict[location] = partial_labels
            else:
                # Modify
                old_boundingboxes = labels_dict[location]
                new_boundingboxes = old_boundingboxes + partial_labels
                labels_dict[location] = new_boundingboxes
            line_count += 1
    print(f'Processed {line_count} lines.')

with open('train.txt','w') as train_file:
    for key in labels_dict:
        train_file.write(key + ' ')
        for label in labels_dict[key]:
            train_file.write(str(label.bounding_box[1]) + ',' + str(label.bounding_box[0]) + ',' + str(label.bounding_box[1] + label.bounding_box[3]) + ',' +  str(label.bounding_box[0] + label.bounding_box[2]) + ',' + str(label.object_class) + ' ')
        train_file.write('\n')