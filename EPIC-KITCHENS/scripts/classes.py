import pandas as pd

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
labels = labels.loc[labels['participant_id'].isin(['P01','P02','P03','P04'])]


dictionary = labels.noun_class.value_counts().to_dict()
{k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}

i = 0
for key, value in dictionary.items():
    if i > 10:
        break
    i += 1
    print(f'Class: {key} Value: {value}')