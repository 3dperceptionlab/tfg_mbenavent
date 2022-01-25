import pandas as pd

nouns = {}
counter = 0
with open('processed-labels/epic-kitchens_classes_full.txt') as f:
    lines = f.readlines()
    for line in lines:
        nouns[line.strip('\n')] = counter
        counter = counter + 1


actions = pd.read_csv('downloaded-labels/EPIC_many_shot_actions.csv')
action_dict = {}    # Noun ID + verbs
nouns_dict = {}     # Noun ID + noun
print(nouns)
for index, action in actions.iterrows():
    if action['noun'] not in nouns:
        continue
    noun_id = nouns[action['noun']]
    if noun_id in action_dict:
        #action_dict[action['noun_class']] = np.append(action_dict[action['noun_class']], action['verb'])
        action_dict[noun_id].append(action['verb'])
    else:
        #action_dict[action['noun_class']] = np.array(action['verb'])
        action_dict[noun_id] = [action['verb']]
        nouns_dict[noun_id] = action['noun']


with open('processed-labels/actions_per_noun-full.csv', 'w') as file:
    file.write('noun_id;noun;verbs\n')
    for key, value in action_dict.items():
        file.write(f'{key};{nouns_dict[key]};{value}\n')
