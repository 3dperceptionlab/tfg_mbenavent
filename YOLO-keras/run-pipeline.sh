python action_estimation.py \
--model_type yolo4_darknet \
--model_type_hand yolo3_darknet \
--weights_path /workspace/Resultados/yolo4_epickitchens_cspdarknet_dumped.h5 \
--weights_path_hands /workspace/Resultados/yolo_egodaily_one_class_2_dumped.h5 \
--classes_path /workspace/tfg_mbenavent/EPIC-KITCHENS/processed-labels/epic-kitchens_classes.txt \
--classes_path_hands /workspace/tfg_mbenavent/EGO-DAILY/egodaily_class_1hand.txt \
--actions_path /workspace/tfg_mbenavent/EPIC-KITCHENS/processed-labels/actions_per_noun.csv \
--anchors_path configs/yolo3_anchors.txt \
--input=/datasets/EPIC-KITCHENS/P01/videos/P01_01.MP4 \
--output=result.mp4

