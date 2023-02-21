import pandas as pd
from os import listdir, path
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from statistics import mean, median

obj_annotation_path = 'ADL_annotations/object_annotation/'
action_annotation_path = 'ADL_annotations/action_annotation/'
output_path = 'new_annotations'
frame_rate = 30
classes = {}
class_counter = 0
instances_per_class = {}

def process_file(filename:str):
    df = pd.read_csv(filename, delimiter=' ', header=None, 
                names=['object_track_id','x_min','y_min','x_max','y_max','frame_number','active','class','NaN'])
    actor_id = int(filename.split('/')[2].split('_')[3].split('.')[0])
    global classes
    global class_counter
    global instances_per_class
    x = []
    y = []
    for index, row in df.iterrows():
        if row['class'] not in classes:
            classes[row['class']] = class_counter
            instances_per_class[row['class']]=0
            class_counter += 1
        instances_per_class[row['class']]+=1
        obj_class = classes[row['class']]
        obj = str(row['x_min'])+','+str(row['y_min'])+','+str(row['x_max'])+','+str(row['y_max'])+','+str(obj_class)
        
        x.append((obj, filename.split('/')[2].split('_')[3].split('.')[0], row['frame_number'], row['class']))
        y.append(obj_class)
    return x, y

def get_annotations():
    x = []
    y = []
    global instances_per_class
    global classes
    global output_path
    obj_annotation_files = listdir(obj_annotation_path)
    for filename in obj_annotation_files:
        if 'frames' not in filename: # Consider only files with name: object_annot_P_XX.txt
            px, py = process_file(path.join(obj_annotation_path, filename))
            assert(len(px)==len(py))
            print(f'File: {filename} Annotated frames: {len(px)}')
            x.extend(px)
            y.extend(py)
    plt.bar(instances_per_class.keys(), instances_per_class.values())
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=90)
    plt.savefig(path.join(output_path,'plot_classes_full.jpg'))
    print(f'Mean: {mean(instances_per_class.values())}') # 2931.5
    print(f'Median: {median(instances_per_class.values())}') # 2931.5
    
    instances_per_class = {k: v for k, v in instances_per_class.items() if v >= 2500}
    plt.clf()
    plt.bar(instances_per_class.keys(), instances_per_class.values())
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(rotation=90)
    plt.savefig(path.join(output_path, 'plot_classes_reduced.jpg'))

    old_classes = {k: v for k, v in classes.items() if k in instances_per_class.keys()}
    count = 0
    classes = {}
    for k in old_classes.keys():
        classes[k] = count
        count += 1

    print(f'Annotated frames before downsampling: {len(x)}')
    new_x = []
    new_y = []
    for i in range(len(y)):
        if y[i] in old_classes.values():
            class_name = list(old_classes.keys())[list(old_classes.values()).index(y[i])]
            new_y.append(classes[class_name])
            obj, actor, frame, class_name_x = x[i]
            assert(class_name==class_name_x)
            new_obj = obj[0:-len(str(old_classes[class_name]))] + str(classes[class_name])
            new_x.append((new_obj, actor, frame, class_name_x))

    return new_x, new_y

def save_classes():
    # Save classes sorted by ID
    with open(path.join(output_path, 'adl_classes.txt'),'w') as f:
        for c in sorted(classes.items(), key=lambda item: item[1]):
            f.write(c[0] + '\n')

def save_split(x, split_type:str):
    # Group by actor and frame
    x_file = {}
    for bb, actor, frame, _ in x:
        name = 'P_' + actor + '/' + str(frame).zfill(6) + '.jpg'
        if name in x_file:
            x_file[name].append(bb)
        else:
            x_file[name] = [bb]
    
    with open(path.join(output_path,split_type) + '.txt','w') as f:
        for key, value in x_file.items():
            f.write('/datasets/ADL/rgb_frames/' + key + ' ' + ' '.join(value) + '\n')

def save_splits(x, y):
    # Split annotations: stratified fashion, shuffle on random_state=42, 20% for test
    x_train, x_test, _, _ = model_selection.train_test_split(x,y,random_state=42, test_size=0.2)
    save_split(x_train, 'train')
    save_split(x_test, 'test')

def get_time_in_frames(time:str):
    time_split = time.split(':')
    minute = int(time_split[0])*frame_rate*60
    second = int(time_split[1])*frame_rate
    return minute+second

def get_actions():
    actions = []
    with open(path.join(action_annotation_path, 'action_list.txt'), 'r') as f:
            lines = f.readlines()
            for l in lines:
                actions.append(l.split('\'')[1])
    with open(path.join(output_path, 'action_list.txt'), 'w') as f:
        f.write('\n'.join(actions))
    

    action_annotation_files = listdir(action_annotation_path)
    actions_per_actor = {}
    for filename in action_annotation_files:
        if 'P_' in filename: # don't open action_list.txt
            df = pd.read_csv(path.join(action_annotation_path, filename), header=None, delimiter=' ', names=['start','end','class','omit'])
            actor_id = int(filename[2:4])
            actions_per_actor[actor_id] = {}
            for index, row in df.iterrows():
                action_class = int(row['class'])-1
                frame_info = (get_time_in_frames(row['start']), get_time_in_frames(row['end']))
                if action_class in actions_per_actor[actor_id]:
                    actions_per_actor[actor_id][action_class].append(frame_info)
                else:
                    actions_per_actor[actor_id][action_class] = [frame_info]
    return actions, actions_per_actor

def save_actions_per_noun(x, actions, actions_per_actor):
    actions_per_noun = {}
    for _, actor, frame, obj_class in x:
        actions_at_actor = actions_per_actor[int(actor)]
        for action, frame_info in actions_at_actor.items():
            for start, end in frame_info:
                if frame >= start and frame <= end:
                    if obj_class not in actions_per_noun:
                        actions_per_noun[obj_class] = set()
                    actions_per_noun[obj_class].add(action)
    with open(path.join(output_path, 'actions_per_noun.csv'),'w') as f:
        f.write('noun_id;noun;verbs\n')
        for noun, action_ids in actions_per_noun.items():
            f.write(str(classes[noun])+';'+noun+';'+str([actions[i] for i in action_ids])+'\n')

def main():
    x, y = get_annotations()
    assert(len(x)==len(y))
    print(f'Objects annotated: {len(x)}')
    save_splits(x,y)
    del y
    save_classes()
    actions, actions_per_actor = get_actions()
    save_actions_per_noun(x, actions, actions_per_actor)
    # Todas las clases tienen acciones asociadas
    

if __name__ == '__main__':
    main()