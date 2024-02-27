FROM tensorflow/tensorflow:1.15.4-gpu
RUN apt update
RUN apt install --assume-yes libgl1-mesa-glx cmake
RUN pip3 install --upgrade pip
RUN pip install keras==2.3.0 pillow matplotlib opencv-python numpy flask imgaug pandas sklearn
#tensorflow_model_optimization