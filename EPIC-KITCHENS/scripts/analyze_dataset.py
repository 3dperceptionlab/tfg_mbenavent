import pandas as pd
import matplotlib.pyplot as plt

# Obtain interactive objects
int_objects_csv = pd.read_csv('EPIC_many_shot_nouns.csv')
interactive_objects = set()
for index, row in int_objects_csv.iterrows():
    interactive_objects.add(row['noun_class'])


labels = pd.read_csv('EPIC_train_object_labels.csv')

labels = labels.loc[labels['noun_class'].isin(interactive_objects)]
labels = labels.loc[labels['participant_id'].isin(['P01','P02','P03','P04'])]



dictionary = labels.noun_class.value_counts().to_dict()

avg = 0
for key, value in dictionary.items():
    avg += value
avg /= len(dictionary)
print(f"The average number of objects per class is {avg} in a total of {len(dictionary)} classes")
newdict = {}
lower_threshold = avg - 500
upper_threshold = avg + 1000
total_labels = 0
for key, value in dictionary.items():
    if value > lower_threshold: # and value < upper_threshold
        newdict[key] = value
        total_labels += value
    
print(f"The total number of labels is {total_labels} in {len(newdict)} classes")
#plt.bar(range(len(newdict.keys())),newdict.values())
print(len(labels)) # 93121
print(len(labels.loc[labels['noun_class'].isin(newdict.keys())])) # 78437
plt.bar(newdict.keys(),newdict.values())
plt.show()


# Total size: 389811
# Only interactive objects: 309727
