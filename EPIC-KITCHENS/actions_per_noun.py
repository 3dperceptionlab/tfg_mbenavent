import pandas as pd

actions = pd.read_csv('downloaded-labels/EPIC_many_shot_actions.csv')
action_dict = {}    # Noun ID + verbs
nouns_dict = {}     # Noun ID + noun
for index, action in actions.iterrows():
    if action['noun_class'] in action_dict:
        #action_dict[action['noun_class']] = np.append(action_dict[action['noun_class']], action['verb'])
        action_dict[action['noun_class']].append(action['verb'])
    else:
        #action_dict[action['noun_class']] = np.array(action['verb'])
        action_dict[action['noun_class']] = [action['verb']]
        nouns_dict[action['noun_class']] = action['noun']


with open('processed-labels/actions_per_noun.csv', 'w') as file:
    file.write('noun_id,noun,verbs\n')
    for key, value in action_dict.items():
        file.write(f'{key},{nouns_dict[key]},{value}\n')
