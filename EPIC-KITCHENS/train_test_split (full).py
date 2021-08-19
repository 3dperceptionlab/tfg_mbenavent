import pandas as pd
from collections import namedtuple
import ast

import sklearn
from sklearn.model_selection import train_test_split



all_labels = pd.read_csv('downloaded-labels/EPIC_train_object_labels.csv')
X = []
y = []
for index, row in all_labels.iterrows():
    X.append(row) # Save label info as X
    y.append(row['noun_class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



def CountFrequency(my_list):
 
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
 
    return freq


train_freq = CountFrequency(y_train)
test_freq = CountFrequency(y_test)


y_train_len = len(y_train)
y_test_len = len(y_test)

for key, value in train_freq.items():
    diff = abs(value*100/y_train_len - test_freq[key]*100/y_test_len)
    if diff > 0.15:
        print(f'Class: {key}, Train value: {value}, Percentage: {value*100/y_train_len}, Test value: {test_freq[key]}, Percentage test: {test_freq[key]*100/y_test_len}')


all_nouns = pd.read_csv('downloaded-labels/EPIC_noun_classes.csv')
nouns_list = []
for index, row in all_nouns.iterrows():
    nouns_list.append(row['class_key']) # Save the class key, the most generic name


# Process labels and classes
class_dict = {}
ClassData = namedtuple('ClassData', 'new_id name')
def process_dataset(labels):
    labels_dict = {}
    LabelData = namedtuple('LabelData', 'object_class bounding_box')
    for label in labels:
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
with open('processed-labels/train.txt(full)','w') as train_file:
    labels_dict = process_dataset(X_train)
    for key in labels_dict:
        train_file.write(key + ' ')
        for label in labels_dict[key]:
            train_file.write(str(label.bounding_box[1]) + ',' + str(label.bounding_box[0]) + ',' + str(label.bounding_box[1] + label.bounding_box[3]) + ',' +  str(label.bounding_box[0] + label.bounding_box[2]) + ',' + str(label.object_class) + ' ')
        train_file.write('\n')

print('-------- SAVING CLASSES IN YOLO FORMAT --------')
sorted_class_dict = {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[1].new_id)} # Inserted sorted but just in case
with open('processed-labels/epic-kitchens_classes(full).txt','w') as class_file:
    for key, value in sorted_class_dict.items():
        class_file.write(value.name + '\n')

with open('processed-labels/test(full).txt','w') as train_file:
    labels_dict = process_dataset(X_test)
    for key in labels_dict:
        train_file.write(key + ' ')
        for label in labels_dict[key]:
            train_file.write(str(label.bounding_box[1]) + ',' + str(label.bounding_box[0]) + ',' + str(label.bounding_box[1] + label.bounding_box[3]) + ',' +  str(label.bounding_box[0] + label.bounding_box[2]) + ',' + str(label.object_class) + ' ')
        train_file.write('\n')

print('-------- COMPLETED --------')
print('train.txt contains subset of training labels for EPIC-KITCHENS in YOLO format')
print('epic-kitchens_classes.txt contains the classes with interactable objects')