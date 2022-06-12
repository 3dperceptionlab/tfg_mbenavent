import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TerminateOnNaN, LambdaCallback

input_shape = (200,200,3)

class VGG16_Pipeline:
    def __init__(self, weights_path):
        self.actions = self.load_actions()
        print(self.actions)
        self.classes = self.load_classes()
        self.model = self.create_model(len(self.classes), input_shape)
        self.model.load_weights(weights_path)

    def load_actions(self):
        df = pd.read_csv('Annotations/mit_indoor-adl.txt', header=None, sep=';')
        locs_actions = {}
        for idx, row in df.iterrows():
            if row[1] == 'All':
                continue # Skip the ones that not filter
            if row[1] not in locs_actions:
                locs_actions[row[1]] = [row[0]]
            else:
                locs_actions[row[1]].append(row[0])
        return locs_actions
            
    def load_classes(self):
        with open('Annotations/classes.txt') as f:
            class_names = f.readlines()
            class_names = [c.strip() for c in class_names]
        return class_names

    def create_model(self, num_classes, input_shape):
        base_model = VGG16(include_top=False, input_shape=input_shape)
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer
        predictions = Dense(num_classes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        return model

    def predict(self, image_path):
        image = Image.open(image_path)
        image = image.resize(input_shape[:2], Image.BICUBIC)
        image = np.array(image)
        # Some images do not fit the required format after resizing
        if image.shape != input_shape:
            return None, None

        image = image.astype(np.float32) / 255.0
        preds = self.model.predict(np.expand_dims(image,axis=0))
        loc = self.classes[np.argmax(preds)]
        print('Location: ' + loc)
        if loc not in self.actions:
            return loc, []
        else:
            return loc, self.actions[loc]


def main():
    a = VGG16_Pipeline('weights_vgg16-mit_indoor.h5')
    while True:
        print(a.predict(input()))


if __name__ == "__main__":
    main()