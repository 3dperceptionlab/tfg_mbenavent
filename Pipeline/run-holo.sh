python action_estimation.py \
--weights_path /workspace/tfg_mbenavent/Pipeline/trained_models/yolo4_adl.h5 \
--weights_path_hands /workspace/tfg_mbenavent/Pipeline/trained_models/yolo4_egodaily_2hands.h5 \
--classes_path /workspace/tfg_mbenavent/ADL/ADL_YOLO_annotations/adl_classes.txt \
--classes_path_hands /workspace/tfg_mbenavent/EGO-DAILY/egodaily_class_2hands.txt \
--actions_path /workspace/tfg_mbenavent/ADL/ADL_YOLO_annotations/actions_per_noun.csv \
--scene_recognition_weights /workspace/tfg_mbenavent/MIT_INDOOR/weights_vgg16-mit_indoor.h5 \
--holo_path /workspace/tfg_mbenavent/Pipeline/holo_test \
--holo_source https://sts107.feratel.co.at/streams/stsstore103/1/15111_63a2e914-d8ffVid.mp4?dcsdesign=feratel4
