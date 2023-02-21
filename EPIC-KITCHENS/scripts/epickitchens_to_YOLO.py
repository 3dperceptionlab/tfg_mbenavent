import csv
import sys
from collections import namedtuple
import ast
import pandas as pd

# Get all classes
all_nouns = pd.read_csv('downloaded-labels/EPIC_noun_classes.csv')
nouns_list = []
for index, row in all_nouns.iterrows():
    nouns_list.append(row['class_key']) # Save the class key, the most generic name


# Get interactive objects (over a 100 times in the action labels)
interactive_nouns = pd.read_csv('downloaded-labels/EPIC_many_shot_nouns.csv')
interactive_objects = set()
for index, row in interactive_nouns.iterrows():
    interactive_objects.add(row['noun_class'])

# Filter labels to be used
labels = pd.read_csv('downloaded-labels/EPIC_train_object_labels.csv')
labels = labels.loc[labels['noun_class'].isin(interactive_objects)]
test_labels = labels.loc[labels['participant_id'].isin(['P05','P06','P07'])]
labels = labels.loc[labels['participant_id'].isin(['P01','P02','P03','P04'])]

# Get the 10 classes with the most labels
dictionary = labels.noun_class.value_counts().to_dict()
{k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}

i = 0
final_objects = set()
for key, value in dictionary.items():
    if i > 10:
        break
    i += 1
    final_objects.add(key)

labels = labels.loc[labels['noun_class'].isin(final_objects)]
test_labels = test_labels.loc[test_labels['noun_class'].isin(final_objects)]

# Process labels and classes
class_dict = {}
ClassData = namedtuple('ClassData', 'new_id name')
def process_dataset(labels):
    labels_dict = {}
    LabelData = namedtuple('LabelData', 'object_class bounding_box')
    for index, label in labels.iterrows():
        location = '/datasets/EPIC-KITCHENS/' + label['participant_id'] + '/object_detection_images/' + label['video_id'] + '/' + str(label['frame']).rjust(10, '0') + '.jpg'
        bounding_boxes = ast.literal_eval(label['bounding_boxes'])
        object_class = label['noun_class'] # Use id, a same class may have different names

        # Save classes in dictionary
        if object_class not in class_dict:
            class_data = ClassData(new_id=len(class_dict), name=nouns_list[object_class]) # Save the first noun name
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
    return labels_dict




print('-------- SAVING LABELS IN YOLO FORMAT --------')
with open('processed-labels/train.txt','w') as train_file:
    labels_dict = process_dataset(labels)
    for key in labels_dict:
        train_file.write(key + ' ')
        for label in labels_dict[key]:
            train_file.write(str(label.bounding_box[1]) + ',' + str(label.bounding_box[0]) + ',' + str(label.bounding_box[1] + label.bounding_box[3]) + ',' +  str(label.bounding_box[0] + label.bounding_box[2]) + ',' + str(label.object_class) + ' ')
        train_file.write('\n')

print('-------- SAVING CLASSES IN YOLO FORMAT --------')
sorted_class_dict = {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[1].new_id)} # Inserted sorted but just in case
with open('processed-labels/epic-kitchens_classes.txt','w') as class_file:
    for key, value in sorted_class_dict.items():
        class_file.write(value.name + '\n')

with open('processed-labels/test.txt','w') as train_file:
    labels_dict = process_dataset(test_labels)
    for key in labels_dict:
        train_file.write(key + ' ')
        for label in labels_dict[key]:
            train_file.write(str(label.bounding_box[1]) + ',' + str(label.bounding_box[0]) + ',' + str(label.bounding_box[1] + label.bounding_box[3]) + ',' +  str(label.bounding_box[0] + label.bounding_box[2]) + ',' + str(label.object_class) + ' ')
        train_file.write('\n')

print('-------- COMPLETED --------')
print('train.txt contains subset of training labels for EPIC-KITCHENS in YOLO format')
print('epic-kitchens_classes.txt contains the classes with interactable objects')