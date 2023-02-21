# Pipeline implementation

This pipeline is based on a Tensorflow-Keras implementation, please refer to the [original implementation](https://github.com/david8862/keras-YOLOv3-model-set) for further details. In this work we have made some modifications for the recognition pipeline.

## Training and evaluation

You can use the script `run-train.sh` to train a model, `run-dump.sh` dumps the trained model and, `run-eval.sh` evaluates the model. You can use the flag `--help` to get further details about the parameters. Refer to the [original implementation](https://github.com/david8862/keras-YOLOv3-model-set) for further details.

## Running the pipeline

The pipeline has some specific parameters depending on your purpose. Here we can see a basic example:

```console
python action_estimation.py \
--model_type yolo4_darknet \
--model_type_hand yolo4_darknet \
--weights_path /workspace/tfg_mbenavent/Pipeline/trained_models/yolo4_adl.h5 \
--weights_path_hands /workspace/tfg_mbenavent/Pipeline/trained_models/yolo4_egodaily_2hands.h5 \
--classes_path /workspace/tfg_mbenavent/ADL/ADL_YOLO_annotations/adl_classes.txt \
--classes_path_hands /workspace/tfg_mbenavent/EGO-DAILY/egodaily_class_2hands.txt \
--actions_path /workspace/tfg_mbenavent/ADL/ADL_YOLO_annotations/actions_per_noun.csv \
--anchors_path configs/yolo3_anchors.txt \
--image
```

### Video input

You can change the `--image` flag by `--input` and `--output` if you want to get the inference on videos.

### Live HoloLens input

The `--holo` flag is used to run the Flask RESTful API used by a HoloLens device to send images and get the inference results. In the [HoloYOLO](https://github.com/3dperceptionlab/HoloYOLO) repository you may find further details about this.
