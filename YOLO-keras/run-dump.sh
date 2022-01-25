#!/bin/bash
python yolo.py \
    --model_type=yolo4_darknet \
    --weights_path=logs/000/trained_final.h5 \
    --anchors_path=configs/yolo3_anchors.txt \
    --classes_path=/workspace/tfg_mbenavent/EGO-DAILY/egodaily_class_2hands.txt \
     --model_image_size=416x416 \
     --dump_model \
     --output_model_file=/workspace/Resultados/yolo4_egodaily_2classes.h5
