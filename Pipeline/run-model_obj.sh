python yolo.py \
--model_type yolo4_darknet \
--weights_path /workspace/tfg_mbenavent/Pipeline/trained_models/yolo4_epickitchens-full_cspdarknet_dumped.h5 \
--classes_path /workspace/tfg_mbenavent/EPIC-KITCHENS/processed-labels/epic-kitchens_classes_full.txt \
--anchors_path configs/yolo3_anchors.txt \
--image
#--input /datasets/ADL/videos/P_11.MP4 \
#--output video.mp4