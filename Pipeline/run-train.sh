#!/bin/bash
python train.py \
	--model_type yolo4_darknet \
	--annotation_file /workspace/tfg_mbenavent/ADL/ADL_YOLO_annotations/train.txt \
	--classes_path /workspace/tfg_mbenavent/ADL/ADL_YOLO_annotations/adl_classes.txt \
	--transfer_epoch=15 \
	--optimizer=adam \
	--batch_size=32 \
	--init_epoch=0 \
	--total_epoch=150 \
	--learning_rate=0.0001 \
	--eval_epoch_interval 5
