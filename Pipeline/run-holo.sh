python action_estimation.py \
--weights_path /workspace/tfg_mbenavent/Pipeline/weights/yolo4_adl.h5 \
--weights_path_hands /workspace/tfg_mbenavent/Pipeline/weights/yolo4_egodaily_2hands.h5 \
--classes_path /workspace/tfg_mbenavent/ADL/ADL_YOLO_annotations/adl_classes.txt \
--classes_path_hands /workspace/tfg_mbenavent/EGO-DAILY/annotations/egodaily_class_2hands.txt \
--actions_path /workspace/tfg_mbenavent/ADL/ADL_YOLO_annotations/actions_per_noun.csv \
--holo
