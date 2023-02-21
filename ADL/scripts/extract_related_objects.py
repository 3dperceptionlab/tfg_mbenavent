import pandas as pd
import ast

# Read file
actions_file = pd.read_csv('ADL_YOLO_annotations/actions_per_noun.csv', delimiter=';')

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
    print(noun)
    print(related)

'''
Conclusiones: 
* Las relaciones son demasiado estrechas por la forma de extraer las relaciones objeto-accion, incluye casi todos los objetos
* Se puede probar reduciendo manualmente las acciones por objeto y volver a ejecutar
'''
