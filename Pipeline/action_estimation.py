#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a YOLOv3/YOLOv2 style detection model on test images.
"""

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
from tensorflow.keras.utils import multi_gpu_model
import pandas as pd
import ast
sys.path.insert(1, '../MIT_INDOOR')
import VGG16_Pipeline


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

optimize_tf_gpu(tf, K)

#tf.enable_eager_execution()

default_config = {
        "model_type": 'tiny_yolo3_darknet',
        "weights_path": os.path.join('weights', 'yolov3-tiny.h5'),
        "pruning_model": False,
        "anchors_path": os.path.join('configs', 'tiny_yolo3_anchors.txt'),
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
            activity_classes = out_classes
        else:
            activity_classes = []
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
                    # Uncomment to print observed surroundings
                    # hand_center = (round(hand_center[0]),round(hand_center[1]))
                    # overlay = image_array.copy()
                    # cv2.circle(overlay, hand_center, round(best_distance + extra_area), (0, 0, 255), -1)
                    # opacity = 0.2
                    # cv2.addWeighted(overlay, opacity, image_array, 1 - opacity, 0, image_array)
                    for distance, obj_class in zip(distances, out_classes):
                        if distance <= best_distance + extra_area:
                            activity_classes.append(obj_class)


            if len(activity_classes)==0:
                activity_classes = out_classes
        image_array = draw_boxes(image_array, hand_out_boxes, hand_out_classes, hand_out_scores, self.hand_classes_names, self.hand_colors)
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors, activity_classes=activity_classes, actions=self.actions)
        
        actions = []
        for obj in activity_classes:
            actions.append(self.actions[obj])
            
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
        return Image.fromarray(image_array), out_boxes, out_classnames, out_scores


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



def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(0 if video_path == '0' else video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    isOutput = True if output_path != "" else False
    if isOutput:
        # here we encode the video to MPEG-4 for better compatibility, you can use ffmpeg later
        # to convert it to x264 to reduce file size:
        # ffmpeg -i test.mp4 -vcodec libx264 -f mp4 test_264.mp4
        #
        #video_FourCC    = cv2.VideoWriter_fourcc(*'XVID') if video_path == '0' else cv2.VideoWriter_fourcc(*"mp4v")
        video_FourCC    = cv2.VideoWriter_fourcc(*"mp4v")
        video_fps       = vid.get(cv2.CAP_PROP_FPS)
        video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        #print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, (5. if video_path == '0' else video_fps), video_size)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    len_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    current_len = 0
    while True:
        is_cap, frame = vid.read()
        if frame is None:
            continue
        current_len += 1
        print("FRAME COUNT: " + str(current_len) + "/" + str(len_frames))
        if len_frames == current_len:
            break
        image = Image.fromarray(frame)
        image, _, _, _ = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        #cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        cv2.waitKey(20)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
    # Release everything if job is finished
    vid.release()
    if isOutput:
        out.release()
    #cv2.destroyAllWindows()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, _, _, _ = yolo.detect_image(image)
            r_image.save('sample_results/adl/' + img.replace('/','-'))


def main():
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='demo or dump out YOLO h5 model')
    '''
    Command line options
    '''
    parser.add_argument(
        '--actions_path', type=str,
        help='Objects-Actions relationship TXT file path'
    )

    parser.add_argument(
        '--model_type', type=str,
        help='YOLO model type: yolo3_mobilenet_lite/tiny_yolo3_mobilenet/yolo3_darknet/..., default ' + YOLO.get_defaults("model_type")
    )

    parser.add_argument(
        '--model_type_hand', type=str,
        help='YOLO model type: yolo3_mobilenet_lite/tiny_yolo3_mobilenet/yolo3_darknet/..., default ' + YOLO.get_defaults("model_type")
    )

    parser.add_argument(
        '--weights_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("weights_path")
    )

    parser.add_argument(
        '--weights_path_hands', type=str,
        help='path to model weight file, default: none'
    )

    parser.add_argument(
        '--pruning_model', default=False, action="store_true",
        help='Whether to be a pruning model/weights file, default ' + str(YOLO.get_defaults("pruning_model"))
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--classes_path_hands', type=str,
        help='path to class definitions, default: none'
    )

    parser.add_argument(
        '--model_image_size', type=str,
        help='model image input size as <height>x<width>, default ' +
        str(YOLO.get_defaults("model_image_size")[0])+'x'+str(YOLO.get_defaults("model_image_size")[1]),
        default=str(YOLO.get_defaults("model_image_size")[0])+'x'+str(YOLO.get_defaults("model_image_size")[1])
    )

    parser.add_argument(
        '--elim_grid_sense', default=False, action="store_true",
        help = "Eliminate grid sensitivity, default " + str(YOLO.get_defaults("elim_grid_sense"))
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )
    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    '''
    Command line positional arguments -- for model dump
    '''
    parser.add_argument(
        '--dump_model', default=False, action="store_true",
        help='Dump out training model to inference model'
    )

    parser.add_argument(
        '--output_model_file', type=str,
        help='output inference model file'
    )

    parser.add_argument(
        '--scene_recognition_weights', default=None,
        help='path for scene recognition weights'
    )

    args = parser.parse_args()
    # param parse
    if args.model_image_size:
        height, width = args.model_image_size.split('x')
        args.model_image_size = (int(height), int(width))
        assert (args.model_image_size[0]%32 == 0 and args.model_image_size[1]%32 == 0), 'model_image_size should be multiples of 32'

    # get wrapped inference object
    yolo = YOLO_np(**vars(args))

    if args.dump_model:
        """
        Dump out training model to inference model
        """
        if not args.output_model_file:
            raise ValueError('output model file is not specified')

        print('Dumping out training model to inference model')
        yolo.dump_model_file(args.output_model_file)
        sys.exit()

    if args.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in args:
            print(" Ignoring remaining command line arguments: " + args.input + "," + args.output)
        detect_img(yolo)
    elif "input" in args:
        detect_video(yolo, args.input, args.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")


if __name__ == '__main__':
    main()
