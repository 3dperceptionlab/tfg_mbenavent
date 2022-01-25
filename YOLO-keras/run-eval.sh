#!/bin/bash
python eval.py \
    --model_path=/workspace/Resultados/yolo4_epickitchens-full_cspdarknet_dumped.h5 \
    --anchors_path=configs/yolo3_anchors.txt \
    --classes_path=/workspace/tfg_mbenavent/EPIC-KITCHENS/processed-labels/epic-kitchens_classes_full.txt \
    --model_image_size=416x416 \
    --eval_type=VOC \
    --iou_threshold=0.5 \
    --conf_threshold=0.001 \
    --annotation_file=/workspace/tfg_mbenavent/EPIC-KITCHENS/processed-labels/test_full.txt \
