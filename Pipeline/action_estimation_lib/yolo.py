import colorsys
import math
import os, sys, argparse
import cv2
import time
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Lambda
from tensorflow_model_optimization.sparsity import keras as sparsity
from PIL import Image

from yolo5.model import get_yolo5_model, get_yolo5_inference_model
from yolo5.postprocess_np import yolo5_postprocess_np
from yolo3.model import get_yolo3_model, get_yolo3_inference_model
from yolo3.postprocess_np import yolo3_postprocess_np
from yolo2.model import get_yolo2_model, get_yolo2_inference_model
from yolo2.postprocess_np import yolo2_postprocess_np
from common.data_utils import preprocess_image
from common.utils import get_classes, get_anchors, get_colors, draw_boxes, optimize_tf_gpu
try:
    from tensorflow.keras.utils import multi_gpu_model
except ImportError:
    from tensorflow.keras.utils.multi_gpu_utils import multi_gpu_model
import pandas as pd
import ast

sys.path.insert(1, '../MIT_INDOOR')
import VGG16_Pipeline

default_config = {
        "model_type": 'yolo4_darknet',
        "model_type_hand": 'yolo4_darknet',
        "weights_path": os.path.join('weights', 'yolov3-tiny.h5'),
        "pruning_model": False,
        "anchors_path": os.path.join('configs', 'yolo3_anchors.txt'),
        "classes_path": os.path.join('configs', 'coco_classes.txt'),
        "score" : 0.2,
        "iou" : 0.4,
        "model_image_size" : (416, 416),
        "elim_grid_sense": False,
        "gpu_num" : 1,
    }

class YOLO_np(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(YOLO_np, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        self.class_names = get_classes(self.classes_path)
        self.hand_classes_names = get_classes(self.classes_path_hands)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)
        self.hand_colors = get_colors(self.hand_classes_names)
        K.set_learning_phase(0)
        self.yolo_model = self._generate_model(os.path.expanduser(self.weights_path), len(self.class_names), self.model_type)
        self.yolo_hand_model = self._generate_model(os.path.expanduser(self.weights_path_hands), len(self.hand_classes_names), self.model_type_hand)
        self.actions = {}
        print('--------------------------------------------------------------')
        print(self.actions_path)
        print('--------------------------------------------------------------')
        actions_file = pd.read_csv(self.actions_path, delimiter=';')
        for index, action in actions_file.iterrows():
            self.actions[int(action['noun_id'])] = ast.literal_eval(action['verbs'])
        if self.scene_recognition_weights is None:
            self.scene_recognition = None
        else:
            self.scene_recognition = VGG16_Pipeline.VGG16_Pipeline(self.scene_recognition_weights)

    def _generate_model(self, weights_path, num_classes, model_type):
        '''to generate the bounding boxes'''
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        #YOLOv3 model has 9 anchors and 3 feature layers but
        #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        #so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors//3

        try:
            if model_type.startswith('scaled_yolo4_') or model_type.startswith('yolo5_'):
                # Scaled-YOLOv4 & YOLOv5 entrance
                yolo_model, _ = get_yolo5_model(model_type, num_feature_layers, num_anchors, num_classes, input_shape=self.model_image_size + (3,), model_pruning=self.pruning_model)
            elif model_type.startswith('yolo3_') or model_type.startswith('yolo4_') or \
                 model_type.startswith('tiny_yolo3_') or model_type.startswith('tiny_yolo4_'):
                # YOLOv3 & v4 entrance
                yolo_model, _ = get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, input_shape=self.model_image_size + (3,), model_pruning=self.pruning_model)
            elif model_type.startswith('yolo2_') or model_type.startswith('tiny_yolo2_'):
                # YOLOv2 entrance
                yolo_model, _ = get_yolo2_model(model_type, num_anchors, num_classes, input_shape=self.model_image_size + (3,), model_pruning=self.pruning_model)
            else:
                raise ValueError('Unsupported model type')

            yolo_model.load_weights(weights_path) # make sure model, anchors and classes match
            if self.pruning_model:
                yolo_model = sparsity.strip_pruning(yolo_model)
            yolo_model.summary()
        except Exception as e:
            print(repr(e))
            assert yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(weights_path))
        if self.gpu_num>=2:
            yolo_model = multi_gpu_model(yolo_model, gpus=self.gpu_num)

        return yolo_model


    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_image_size)
        #origin image shape, in (height, width) format
        image_shape = tuple(reversed(image.size))

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape, self.model_type, self.yolo_model, self.class_names)
        print('Found {} boxes for {}'.format(len(out_boxes), 'object'))
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))

        start = time.time()
        current_scene = None
        actions_at_location = None
        if self.scene_recognition is not None:
            current_scene, actions_at_location = self.scene_recognition.predict(image)
            print("Found location: " + current_scene)
            print("Inference time: {:.8f}s".format(end - start))
        end = time.time()

        start = time.time()
        hand_out_boxes, hand_out_classes, hand_out_scores = self.predict(image_data, image_shape,self.model_type_hand, self.yolo_hand_model, self.hand_classes_names)
        print('Found {} boxes for {}'.format(len(hand_out_boxes), 'hand'))
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))

        #draw result on input image
        image_array = np.array(image, dtype='uint8')

        if len(hand_out_boxes) == 0:
            indexes_of_interest = [*range(len(out_boxes))] # All the objects IDs
        else:
            indexes_of_interest = []
            # Get closest object 
            for hand_bb in hand_out_boxes:
                xmin, ymin, xmax, ymax = map(int, hand_bb)
                hand_center = ((xmin+(xmax-xmin)/2),(ymin+(ymax-ymin)/2))
                # Distance between hand_center a top-right corner
                extra_area = math.sqrt((hand_center[0]-xmin)**2+(hand_center[1]-ymin)**2)
                best_distance = -1
                distances = []
                for obj_bb, obj_class in zip(out_boxes,out_classes):
                    xmin, ymin, xmax, ymax = map(int, obj_bb)
                    obj_center = ((xmin+(xmax-xmin)/2),(ymin+(ymax-ymin)/2))
                    distance = math.sqrt((hand_center[0]-obj_center[0])**2+(hand_center[1]-obj_center[1])**2)
                    distances.append(distance)
                    if best_distance == -1 or distance < best_distance:
                        best_distance = distance

                if best_distance != -1:
                    '''
                    Uncomment to print observed surroundings
                    '''
                    # hand_center = (round(hand_center[0]),round(hand_center[1]))
                    # overlay = image_array.copy()
                    # cv2.circle(overlay, hand_center, round(best_distance + extra_area), (0, 0, 255), -1)
                    # opacity = 0.2
                    # cv2.addWeighted(overlay, opacity, image_array, 1 - opacity, 0, image_array)
                    for index, distance in enumerate(distances):
                        if distance <= best_distance + extra_area:
                            indexes_of_interest.append(index)


            if len(indexes_of_interest)==0:
                indexes_of_interest = [*range(len(out_boxes))]
        image_array = draw_boxes(image_array, hand_out_boxes, hand_out_classes, hand_out_scores, self.hand_classes_names, self.hand_colors)
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors, indexes_of_interest=indexes_of_interest, actions=self.actions)
        
        # Obtain global action set
        actions = []
        classes_of_interest = list(map(lambda x: out_classes[x], indexes_of_interest))

        for class_of_interest in classes_of_interest:
            actions.append(self.actions[class_of_interest])
            
        # Intersection of actions
        final_actions = actions[0] if len(actions)>0 else []
        if actions_at_location is None:
            actions_at_location = final_actions
        for act in actions[1:]:
            final_actions = list(set(final_actions) & set(act) & set(actions_at_location))

        # Print most likely actions
        text = ((current_scene + ':') if (current_scene is not None) else '') + str(final_actions)
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.5 #4.5
        thickness = 2 #3 #6
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
        padding = 7 # 15
        rect_height = text_height + padding * 3
        height = image_array.shape[0]
        # Create area for text
        image_array = cv2.copyMakeBorder(image_array, 0, rect_height, 0, 0, cv2.BORDER_CONSTANT, None, [0, 0, 0])

        # Print actions
        cv2.putText(image_array, text, (0 + padding, height + text_height + padding*2), font,
                fontScale=font_scale,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA,
                thickness=thickness)

        out_classnames = [self.class_names[c] for c in out_classes]

        # Create json of most relevant objects
        relevant_objects = []
        for idx in indexes_of_interest:
            obj = {}
            obj['class'] = self.class_names[out_classes[idx]]
            obj['actions'] = self.actions[out_classes[idx]]
            xmin, ymin, xmax, ymax = map(int, out_boxes[idx])
            obj['xmin'] = xmin
            obj['ymin'] = ymin
            obj['xmax'] = xmax
            obj['ymax'] = ymax
            relevant_objects.append(obj)


        return Image.fromarray(image_array), out_boxes, out_classnames, out_scores, (relevant_objects, final_actions)


    def predict(self, image_data, image_shape, model_type, yolo_model, class_names):
        num_anchors = len(self.anchors)
        if model_type.startswith('scaled_yolo4_') or model_type.startswith('yolo5_'):
            # Scaled-YOLOv4 & YOLOv5 entrance, enable "elim_grid_sense" by default
            out_boxes, out_classes, out_scores = yolo5_postprocess_np(yolo_model.predict(image_data), image_shape, self.anchors, len(class_names), self.model_image_size, max_boxes=100, confidence=self.score, iou_threshold=self.iou, elim_grid_sense=True)
        elif model_type.startswith('yolo3_') or model_type.startswith('yolo4_') or \
             model_type.startswith('tiny_yolo3_') or model_type.startswith('tiny_yolo4_'):
            # YOLOv3 & v4 entrance
            out_boxes, out_classes, out_scores = yolo3_postprocess_np(yolo_model.predict(image_data), image_shape, self.anchors, len(class_names), self.model_image_size, max_boxes=100, confidence=self.score, iou_threshold=self.iou, elim_grid_sense=self.elim_grid_sense)
        elif model_type.startswith('yolo2_') or model_type.startswith('tiny_yolo2_'):
            # YOLOv2 entrance
            out_boxes, out_classes, out_scores = yolo2_postprocess_np(yolo_model.predict(image_data), image_shape, self.anchors, len(class_names), self.model_image_size, max_boxes=100, confidence=self.score, iou_threshold=self.iou, elim_grid_sense=self.elim_grid_sense)
        else:
            raise ValueError('Unsupported model type')

        return out_boxes, out_classes, out_scores


    def dump_model_file(self, output_model_file):
        self.yolo_model.save(output_model_file)



class YOLO(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        optimize_tf_gpu(tf, K)
        super(YOLO, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)
        K.set_learning_phase(0)

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #YOLOv3 model has 9 anchors and 3 feature layers but
        #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        #so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors//3

        if self.model_type.startswith('scaled_yolo4_') or self.model_type.startswith('yolo5_'):
            # Scaled-YOLOv4 & YOLOv5 entrance, enable "elim_grid_sense" by default
            inference_model = get_yolo5_inference_model(self.model_type, self.anchors, num_classes, weights_path=weights_path, input_shape=self.model_image_size + (3,), confidence=self.score, iou_threshold=self.iou, elim_grid_sense=True)
        elif self.model_type.startswith('yolo3_') or self.model_type.startswith('yolo4_') or \
             self.model_type.startswith('tiny_yolo3_') or self.model_type.startswith('tiny_yolo4_'):
            # YOLOv3 & v4 entrance
            inference_model = get_yolo3_inference_model(self.model_type, self.anchors, num_classes, weights_path=weights_path, input_shape=self.model_image_size + (3,), confidence=self.score, iou_threshold=self.iou, elim_grid_sense=self.elim_grid_sense)
        elif self.model_type.startswith('yolo2_') or self.model_type.startswith('tiny_yolo2_'):
            # YOLOv2 entrance
            inference_model = get_yolo2_inference_model(self.model_type, self.anchors, num_classes, weights_path=weights_path, input_shape=self.model_image_size + (3,), confidence=self.score, iou_threshold=self.iou, elim_grid_sense=self.elim_grid_sense)
        else:
            raise ValueError('Unsupported model type')

        inference_model.summary()
        return inference_model

    def predict(self, image_data, image_shape):
        out_boxes, out_scores, out_classes = self.inference_model.predict([image_data, image_shape])

        out_boxes = out_boxes[0]
        out_scores = out_scores[0]
        out_classes = out_classes[0]

        out_boxes = out_boxes.astype(np.int32)
        out_classes = out_classes.astype(np.int32)
        return out_boxes, out_classes, out_scores

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_image_size)

        # prepare origin image shape, (height, width) format
        image_shape = np.array([image.size[1], image.size[0]])
        image_shape = np.expand_dims(image_shape, 0)

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        end = time.time()
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        print("Inference time: {:.8f}s".format(end - start))

        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)

        out_classnames = [self.class_names[c] for c in out_classes]
        return Image.fromarray(image_array), out_boxes, out_classnames, out_scores

    def dump_model_file(self, output_model_file):
        self.inference_model.save(output_model_file)

    def dump_saved_model(self, saved_model_path):
        model = self.inference_model
        os.makedirs(saved_model_path, exist_ok=True)

        tf.keras.experimental.export_saved_model(model, saved_model_path)
        print('export inference model to %s' % str(saved_model_path))