import pandas as pd
import numpy as np

nouns = {}
counter = 0
with open('processed-labels/epic-kitchens_classes_full.txt') as f:
    lines = f.readlines()
    for line in lines:
        nouns[line.strip('\n')] = counter
        counter = counter + 1

train = pd.read_csv('downloaded-labels/EPIC_train_action_labels.csv')
action_counts = train.apply(lambda r: (r['verb_class'], r['noun_class']), axis=1).value_counts()
actions_in_training = set(action_counts.index.values) # {(verb_class, noun_class)}

nouns_or = pd.read_csv('downloaded-labels/EPIC_noun_classes.csv')
nouns_or = {rows['noun_id']:rows['class_key'] for idx,rows in nouns_or.iterrows()}

verbs = pd.read_csv('downloaded-labels/EPIC_verb_classes.csv')
verbs = {rows['verb_id']:rows['class_key'] for idx,rows in verbs.iterrows()}

# dict -> noun_id (original) : verbs list
actions_per_noun_or_id = {}
for tuple in actions_in_training:
    if tuple[1] in actions_per_noun_or_id:
        actions_per_noun_or_id[tuple[1]].append(verbs[tuple[0]])
    else:
        actions_per_noun_or_id[tuple[1]] = [verbs[tuple[0]]]


with open('processed-labels/actions_per_noun-full.csv', 'w') as file:
    file.write('noun_id;noun;verbs\n')
    for key,value in actions_per_noun_or_id.items():
        noun_key = nouns_or[key]
        if noun_key in nouns:
            noun_id = nouns[noun_key]
            file.write(f'{noun_id};{noun_key};{value}\n')

# action_dict = {}    # Noun ID + verbs
# nouns_dict = {}     # Noun ID + noun
# print(nouns)
# for index, action in actions.iterrows():
#     if action['noun'] not in nouns:
#         continue
#     noun_id = nouns[action['noun']]
#     if noun_id in action_dict:
#         #action_dict[action['noun_class']] = np.append(action_dict[action['noun_class']], action['verb'])
#         action_dict[noun_id].append(action['verb'])
#     else:
#         #action_dict[action['noun_class']] = np.array(action['verb'])
#         action_dict[noun_id] = [action['verb']]
#         nouns_dict[noun_id] = action['noun']


# with open('processed-labels/actions_per_noun-full.csv', 'w') as file:
#     file.write('noun_id;noun;verbs\n')
#     for key, value in action_dict.items():
#         file.write(f'{key};{nouns_dict[key]};{value}\n')