#!/bin/bash
python eval.py \
    --model_path=/workspace/Resultados/yolo4_adl_2.h5 \
    --anchors_path=configs/yolo3_anchors.txt \
    --classes_path=/workspace/tfg_mbenavent/ADL/ADL_YOLO_annotations/adl_classes.txt \
    --model_image_size=416x416 \
    --eval_type=VOC \
    --iou_threshold=0.5 \
    --conf_threshold=0.1 \
    --annotation_file=/workspace/tfg_mbenavent/ADL/ADL_YOLO_annotations/test.txt
