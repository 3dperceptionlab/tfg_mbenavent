#!/bin/bash
python yolo.py \
    --model_type=yolo4_darknet \
    --weights_path=logs/000/trained_final.h5 \
    --anchors_path=configs/yolo3_anchors.txt \
    --classes_path=/workspace/tfg_mbenavent/ADL/ADL_YOLO_annotations/adl_classes.txt \
    --model_image_size=416x416 \
    --dump_model \
    --output_model_file=trained_models/yolo4_adl_4jun.h5
