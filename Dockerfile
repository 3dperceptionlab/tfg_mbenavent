FROM tensorflow/tensorflow:1.6.0-gpu-py3
RUN apt update
#RUN apt install --assume-yes python3-pip
RUN apt install --assume-yes libgl1-mesa-glx
RUN pip3 install --upgrade pip
RUN pip3 install keras==2.1.5 pillow matplotlib opencv-python numpy