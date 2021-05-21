python action_estimation.py \
--model_type yolo3_darknet \
--model_type_hand yolo3_darknet \
--weights_path /workspace/Resultados/yolo_epickitchens_darknet_dumped.h5 \
--weights_path_hands /workspace/Resultados/yolo_egodaily_one_class_2_dumped.h5 \
--classes_path /workspace/tfg_mbenavent/EPIC-KITCHENS/processed-labels/epic-kitchens_classes.txt \
--classes_path_hands /workspace/tfg_mbenavent/EGO-DAILY/egodaily_class_1hand.txt \
--anchors_path configs/yolo3_anchors.txt \
--image