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

batch_size = 32
val_split = 0.2
input_shape = (200,200,3)
epochs = 250
arg_train = False
arg_test = True
arg_predict = False

def create_model(num_classes, input_shape=input_shape):
    base_model = VGG16(weights='weights_vgg16-places365.h5', include_top=False, input_shape=input_shape)
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

def load_images(paths_list,input_shape=input_shape):
    images = []
    removed_indexes = []
    for idx, img_path in enumerate(paths_list):
        image = Image.open(img_path)
        image = image.resize(input_shape[:2], Image.BICUBIC)
        image = np.array(image)
        # Some images do not fit the required format after resizing
        if image.shape != input_shape:
            removed_indexes.append(idx)
            continue
        image = image.astype(np.float32) / 255.0
        images.append(image)
    return np.array(images), removed_indexes

def load_annotations(path, num_classes):
    df = pd.read_csv(path, header=None)
    X = df[0].tolist()
    y = df[1].tolist()
    X, removed_indexes = load_images(X)
    for idx in removed_indexes:
        y.pop(idx)
    y = to_categorical(y, num_classes)
    return X,y

def load_classes():
    with open('Annotations/classes.txt') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def create_splits(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test

def get_data_generator(X_train, y_train, num_train, batch_size):
    datagen = image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=val_split)
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, seed=42, subset="training")
    val_generator = datagen.flow(X_train, y_train, batch_size=batch_size, seed=42, subset="validation")
    return train_generator, val_generator

def train(X_train,y_train,num_classes,model,num_train,num_val):
    # callbacks for training process
    log_dir = 'logs'
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode='min', patience=10, verbose=1, cooldown=0, min_lr=1e-10)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min')
    terminate_on_nan = TerminateOnNaN()
    callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan]

    train_generator, val_generator = get_data_generator(X_train, y_train, num_train, batch_size)


    model.fit_generator(train_generator,
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=val_generator,
            validation_steps=max(1, num_val//batch_size),
            epochs=epochs,
            max_queue_size=10,
            callbacks=callbacks)
    model.save(os.path.join(log_dir, 'fine_tuning_final.h5'))

def test(model, X_test, y_test):
    model.load_weights('logs/fine_tuning_final.h5')
    loss, acc = model.evaluate(X_test,y_test)
    print("Model accuracy: {:5.2f}%".format(100 * acc))

def predict(model, class_names):
    model.load_weights('logs/fine_tuning_final.h5')
    while True:
        print("Image path: ")
        path = input()
        image = Image.open(path)
        image = image.resize(input_shape[:2], Image.BICUBIC)
        image = np.array(image)
        # Some images do not fit the required format after resizing
        if image.shape != input_shape:
            print("Please provide a valid image")
            continue
        image = image.astype(np.float32) / 255.0
        preds = model.predict(np.expand_dims(image,axis=0))
        print(preds)
        # value = np.amax(preds,axis=0)
        # class_idx = np.argmax(preds,axis=0)
        # print(f"{class_idx} {value}%")
            # /home/mbenavent/workspace/tfg_mbenavent/Pipeline/sample_results/sample3.jpg

def main():
    class_names = load_classes()
    num_classes = len(class_names)
    if arg_train or arg_test:
        X,y = load_annotations('Annotations/mit_indoor_adl.txt', num_classes)
        X_train, X_test, y_train, y_test = create_splits(X,y)
        num_val = int(len(X_train)*val_split)
        num_train = len(X_train) - num_val
    model = create_model(num_classes)

    if arg_train:
        train(X_train,y_train,num_classes,model,num_train,num_val)
    if arg_test:
        test(model, X_train, y_train)
    if arg_predict:
        predict(model, class_names)
    

if __name__ == "__main__":
    main()