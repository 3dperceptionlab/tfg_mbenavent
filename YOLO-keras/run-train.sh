#!/bin/bash
python train.py \
	--model_type yolo4_darknet \
	--annotation_file /workspace/tfg_mbenavent/EGO-DAILY/train_2hands.txt \
	--classes_path /workspace/tfg_mbenavent/EGO-DAILY/egodaily_class_2hands.txt \
	--transfer_epoch=15 \
	--optimizer=adam \
	--batch_size=32 \
	--init_epoch=0 \
	--total_epoch=130 \
	--learning_rate=0.00001 \
	--eval_epoch_interval 5
