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
from PIL import Image, UnidentifiedImageError
from action_estimation_lib.yolo import YOLO, YOLO_np
from VideoStream import VideoStream
import hashlib
import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify
import queue
import threading
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

processing_ids = set()
holo_frame_queue = queue.Queue()
holo_results = {}

def worker(yolo):
    while True:
        id, frame = holo_frame_queue.get()
        r_image, _, _, _, r_holo = yolo.detect_image(frame)
        # r_relevant_objects, r_action_intersection = r_holo
        print(f'Processing id: {id}')
        holo_results[id] = r_holo
        processing_ids.remove(id)
        r_image.save("tmp.png")
        holo_frame_queue.task_done()


app = Flask(__name__)
@app.route('/', methods=["GET"])
def home():
    return "<p>POST image on /holo_frame</p> <p> Body (form-data): 'id' (string) and 'img' (string, image in base64 encoding) </p> <p>GET result on returned URL (/holo_result/<id>)</p> <p> Return json including at least 'msg', also 'data' if status code is 200</p>"

@app.route('/holo_frame', methods=["POST"])
def holo_frame():
    if request.form.keys() != {'id','img'}:
        return jsonify({'msg':'invalid form-data'}), 400
    id = request.form['id']
    if id in processing_ids:
        return jsonify({'msg':'non valid id'}), 400
    try:
        im = Image.open(BytesIO(base64.b64decode(request.form['img']))).convert(mode='RGB')
    except UnidentifiedImageError:
        print('Controlled ERROR: UnidentifiedImageError')
        return jsonify({'msg':'non valid image data'}), 400

    holo_frame_queue.put((id,im))
    processing_ids.add(id)
    return jsonify({'msg':'success', 'get':f'{request.url_root}holo_result/{id}'}), 202

@app.route('/holo_result/<id>', methods=["GET"])
def holo_result(id):
    if id in holo_results:
        data, action_intersection = holo_results[id]
        del holo_results[id]
        return jsonify({'msg':'success', 'data': data, 'actions': action_intersection}), 200
    elif id in processing_ids:
        return jsonify({'msg':'in progress'}), 503 # Try again later
    else:
        return jsonify({'msg':'id does not exist'}), 404




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
        image, _, _, _, _ = yolo.detect_image(image)
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


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, _, _, _, _ = yolo.detect_image(image)
            r_image.save('sample_results/adl/' + img.replace('/','-'))


# def detect_holo(yolo, holo_path, source, holo_json):
#     # if img_mode := ('png' or 'jpg' in source): # Python 3.8
#     if 'png' in source or 'jpg' in source:
#         img_mode = True
#         checksum = ''
#         print(f"Reading {source} in {holo_path}")
#     elif 'api' == source:
#         pass
#     else:
#         img_mode = False
#         print(f"Obtaining video from {source}")
#         vStream = VideoStream(source).start()
#     while True:
#         try:
#             if img_mode:
#                 frame = Image.open(os.path.join(holo_path, source))
#                 new_checksum = hashlib.md5(frame.tobytes()).hexdigest()
#                 if checksum==new_checksum:
#                     print("Image not modified, sleeping for 0.5 seconds.")
#                     time.sleep(0.5)
#                     continue
#                 checksum = new_checksum
#             else:
#                 print("Obtaining frame...")
#                 frame = Image.fromarray(vStream.read())
#         except:
#             print("Open Error! Sleeping for 0.5 seconds.")
#             time.sleep(0.5)
#         else:
#             r_image, _, _, _, r_relevant_objects = yolo.detect_image(frame)
#             r_image.save(os.path.join(holo_path, 'result.png'))
#             with open(os.path.join(holo_path, holo_json), 'w') as f:
#                 f.write(json.dumps(r_relevant_objects, indent=4))

def main():
    global yolo_global
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

    parser.add_argument(
        '--holo', default=False, action="store_true",
        help='HoloLens API detection mode'
    )

    # parser.add_argument(
    #     '--holo_path', nargs='?', type=str, required=False, default=None,
    #     help='HoloLens detection mode. Specify path for input/output files.')

    # parser.add_argument(
    #     '--holo_source', nargs='?', type=str, required=False, default='frame.png',
    #     help='Source for YOLO. Either image (png or jpg) name in --holo_path or URL.')

    # parser.add_argument(
    #     '--holo_json', nargs='?', type=str, required=False, default='frame_result.json',
    #     help='JSON file name in --holo_path.')

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
    elif args.holo:
        # threading.Thread(target=lambda: worker(yolo), daemon=True).start()
        # app.run(debug=True, use_reloader=False)
        threading.Thread(target= lambda: app.run(debug=True, use_reloader=False, host="0.0.0.0"), daemon=True).start()
        worker(yolo)


    elif "input" in args:
        detect_video(yolo, args.input, args.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")


if __name__ == '__main__':
    main()
