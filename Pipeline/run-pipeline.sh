python action_estimation.py \
--model_type yolo4_darknet \
--model_type_hand yolo4_darknet \
--weights_path /workspace/Resultados/yolo4_epickitchens-full_cspdarknet_dumped.h5 \
--weights_path_hands /workspace/Resultados/yolo4_egodaily_2classes.h5 \
--classes_path /workspace/tfg_mbenavent/EPIC-KITCHENS/processed-labels/epic-kitchens_classes_full.txt \
--classes_path_hands /workspace/tfg_mbenavent/EGO-DAILY/egodaily_class_2hands.txt \
--actions_path /workspace/tfg_mbenavent/EPIC-KITCHENS/processed-labels/actions_per_noun-full.csv \
--anchors_path configs/yolo3_anchors.txt \
--image
