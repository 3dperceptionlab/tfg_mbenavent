import csv
import sys
from collections import namedtuple
import ast
import pandas as pd

# Get all classes
all_nouns = pd.read_csv('EPIC_noun_classes.csv')
nouns_dict = []
for index, row in all_nouns.iterrows():
    nouns_dict.append(row['class_key']) # Save the class key, the most generic name


# Get interactive objects
nouns = pd.read_csv('EPIC_many_shot_nouns.csv')
interactive_objects = set()
for index, row in nouns.iterrows():
    interactive_objects.add(row['noun_class'])

# Filter labels to be used
labels = pd.read_csv('EPIC_train_object_labels.csv')
labels = labels.loc[labels['noun_class'].isin(interactive_objects)]
labels = labels.loc[labels['participant_id'].isin(['P01','P02','P03','P04'])]

# Process labels and classes
labels_dict = {}
class_dict = {}
LabelData = namedtuple('LabelData', 'object_class bounding_box')
ClassData = namedtuple('ClassData', 'new_id name')
for index, label in labels.iterrows():
    location = '/epic-kitchens/' + label['participant_id'] + '/object_detection_images/' + label['video_id'] + '.tar'
    img_name = './' + str(label['frame']).rjust(10, '0') + '.jpg'
    location = location + ' ' + img_name
    bounding_boxes = ast.literal_eval(label['bounding_boxes'])
    object_class = label['noun_class'] # Use id, a same class may have different names

    # Save classes in dictionary
    if object_class not in class_dict:
        class_data = ClassData(new_id=len(class_dict), name=nouns_dict[object_class]) # Save the first noun name
        class_dict[object_class] = class_data
        object_class = class_data.new_id # Save new_id to print into file
    else:
        object_class = class_dict[object_class].new_id

    # Save labels in dictionary

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

print('-------- SAVING CLASSES IN YOLO FORMAT --------')
sorted_class_dict = {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[1].new_id)} # Inserted sorted but just in case
with open('epic-kitchens_classes.txt','w') as class_file:
    for key, value in sorted_class_dict.items():
        class_file.write(value.name + '\n')


print('-------- SAVING LABELS IN YOLO FORMAT --------')
with open('train.txt','w') as train_file:
    for key in labels_dict:
        train_file.write(key + ' ')
        for label in labels_dict[key]:
            train_file.write(str(label.bounding_box[1]) + ',' + str(label.bounding_box[0]) + ',' + str(label.bounding_box[1] + label.bounding_box[3]) + ',' +  str(label.bounding_box[0] + label.bounding_box[2]) + ',' + str(label.object_class) + ' ')
        train_file.write('\n')

# print('-------- PROCESSING LABELS --------')
# labels_dict = {}
# LabelData = namedtuple("LabelData", "object_class bounding_box")
# with open(sys.argv[2]) as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         else:
#             location = '/epic-kitchens/' + row[2] + '/rgb_frames/' + row[3] + '.tar'
#             img_name = './frame_' + row[4].rjust(10, '0') + '.jpg'
#             location = location + ' ' + img_name
#             bounding_boxes = ast.literal_eval(row[5])
#             object_class = row[0] # Use id, a same class may have different names
#             partial_labels = []
#             for bb in bounding_boxes:
#                 label = LabelData(object_class=object_class, bounding_box=bb)
#                 partial_labels.append(label)
#             if len(bounding_boxes) == 0:
#                 continue
#             if location not in labels_dict:
#                 # Add
#                 labels_dict[location] = partial_labels
#             else:
#                 # Modify
#                 old_boundingboxes = labels_dict[location]
#                 new_boundingboxes = old_boundingboxes + partial_labels
#                 labels_dict[location] = new_boundingboxes
#             line_count += 1
#     print(f'Processed {line_count} epic-kicthens labels.')




# print('-------- COMPLETED --------')
# print('epic-kitchens_classes.txt contains the classes that should be placed in the model folder.')
# print('train.txt contains the labels for training.')