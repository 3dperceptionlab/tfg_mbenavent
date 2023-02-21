import pandas as pd
import ast

# Read file
actions_file = pd.read_csv('processed-labels/actions_per_noun-full.csv', delimiter=';')

# Import actions and nouns
actions = {}
for index, action in actions_file.iterrows():
    actions[(int(action['noun_id']), action['noun'])] = ast.literal_eval(action['verbs'])


related_objects = {}
for (id, noun) in actions.keys():
    related = set()
    for act1 in actions[(id, noun)]:
        for pair in actions.keys():
            if pair[0] == id:
                continue
            for act2 in actions[pair]:
                if act1==act2:
                    related.add(pair[0])
                    break
    related_objects[(id, noun)] = related
    # print(noun)
    # print(related)

total = 0
# with open('out.txt','w') as f:
for key, value in actions.items():
    total += len(value)
        # f.write(key[1] + " " + str(len(value)) + "\n")

print(f'Media de relaciones de objetos: {total//len(actions)}')
print(f'Total objetos: {len(actions)}')
print(f'Proporción relacionados: {(total//len(actions))/len(actions)*100} %')

