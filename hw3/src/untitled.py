'''
To run on GPU, make sure tensorflow-gpu is installed (preferably the same version as tensorflow). 
Tensorflow 1.15.0 tested. Not sure if makes for >= 2.0
'''

flags = {
    'input_snippet_training_dir': '/projectnb/saenkog/awong1/dataset/kitti/processed3/training',
    'input_snippet_validation_dir': '/projectnb/saenkog/awong1/dataset/kitti/processed3/validation',
    'input_snippet_test_dir': '/projectnb/saenkog/awong1/dataset/kitti/processed3/testing',	
    'NUM_EPOCHS': 80,
    'STEPS_PER_EPOCH': 500, # steps_per_epoch = ceil(num_samples / batch_size)
    'INPUT_SHAPE': (128, 128, 1),
    'SEED': 123,
    'SAVE_DIRECTORY': 'conv_mil',
    'MODEL_NAME': 'model33_kitti_80epoch_0.001lr.h5',
    'TRAIN': 1,
    'VISUALIZE': 1,
    'LEARNING_RATE': 0.001,
    'NUM_CLASSES': 2
}

print("c0 | flags")
print(flags)

import os
import argparse

import tensorflow as tf
print("c1 | tf test gpu avail:", tf.test.is_gpu_available()) # True/False
from tensorflow.python.client import device_lib
print("c1 | device_lib.list_local_devices: ", device_lib.list_local_devices()) # list of DeviceAttributes

import keras
import numpy as np

from keras import backend as K
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Layer
from keras.models import Sequential, load_model

from datasets import Mnist
from keras import optimizers


# Neural Network Class
class NoisyAnd(Layer):
    """Custom NoisyAND layer from the Deep MIL paper"""

    def __init__(self, output_dim=flags['NUM_CLASSES'], **kwargs):
        self.output_dim = output_dim
        super(NoisyAnd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = 10  # fixed, controls the slope of the activation
        self.b = self.add_weight(name='b',
                                 shape=(1, input_shape[3]),
                                 initializer='uniform',
                                 trainable=True)
        super(NoisyAnd, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        res = (tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)) / (
                tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]


def define_model(input_shape, num_classes):
    """Define Deep FCN for MIL, layer-by-layer from original paper"""
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(1000, (3, 3), activation='relu'))
    model.add(Conv2D(num_classes, (1, 1), activation='relu'))
    model.add(NoisyAnd(num_classes))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_using_generator(train_generator, val_generator, epochs, steps_per_epoch, seed, input_shape, num_classes):
    np.random.seed(seed)
    model = define_model(input_shape, num_classes)
    model.summary()
    sgd = optimizers.SGD(lr=flags['LEARNING_RATE'])
    print("Learning rate={}".format(flags['LEARNING_RATE']))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, shuffle=True)
    return model

# Database Generator
from keras_preprocessing.image import ImageDataGenerator
class kitti_flow_from_dir_generator():
    def __init__(self, dataset_path):
        self.num_classes = 2
        self.input_shape = (128, 128, 1)
        
        datagen = ImageDataGenerator()
        self.train_generator = datagen.flow_from_directory(
            directory = dataset_path,
            target_size=(128, 128),
            color_mode="grayscale",
            batch_size=128,
            class_mode="categorical",
            shuffle=True,
            seed=123
        )

# Database Loader
def read_and_process_img(img_path, resize_dim=(128, 128)):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, resize_dim)
    img = np.expand_dims(img, axis=4)
    return img

def build_dataset(dataset_path, debug=1):
    X, y = [], []
    for class_name in sorted(os.listdir(dataset_path)):
        # Traverse image dir
        for img_name in sorted(os.listdir(dataset_path + "/" + class_name)):
            img_path = dataset_path + "/" + class_name + "/" + img_name
            img = read_and_process_img(img_path, resize_dim=(128, 128))
            X.append(img)
            y.append(class_name)
    X,y = np.array(X), np.array(y)
    return X, y

# Execution
def main():
    # Create generator
    train_generator = kitti_flow_from_dir_generator(flags['input_snippet_training_dir']).train_generator
    val_generator = kitti_flow_from_dir_generator(flags['input_snippet_validation_dir']).train_generator

    # Make save directory if it doesn't exist
    if not os.path.exists(flags['SAVE_DIRECTORY']):
        os.makedirs(flags['SAVE_DIRECTORY'])

    filepath = os.path.join(flags['SAVE_DIRECTORY'], flags['MODEL_NAME'])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False	# 11,13
    config.gpu_options.per_process_gpu_memory_fraction = 0.1	# 12
    config.log_device_placement = True
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            K.set_session(sess)

            # Train or load model
            if flags['TRAIN']:            
                model = train_using_generator(train_generator, 
                                              val_generator,
                                              epochs=flags['NUM_EPOCHS'],
                                              steps_per_epoch=flags['STEPS_PER_EPOCH'],
                                              seed=flags['SEED'],
                                              input_shape=flags['INPUT_SHAPE'],
                                              num_classes=flags['NUM_CLASSES'])
                model.save(filepath)
                print("Done. Model saved.")
            else:
                model = load_model(filepath, custom_objects={'NoisyAnd': NoisyAnd})

            # Load test set
            X_test, y_test = build_dataset(flags['input_snippet_test_dir'], output_orig=False)
            X_test_orig, _ = build_dataset(flags['input_snippet_test_dir'], output_orig=True)

            print("X_test shape={}, y_test shape={}".format(X_test.shape, y_test.shape))
            y_test_proc = (y_test != 'has_pedestrian')
            print("y_test_proc shape:", y_test_proc.shape)
            # Evaluation
            y_test_one_hot = keras.utils.to_categorical(y_test_proc, num_classes=flags['NUM_CLASSES'], dtype='int')
            print(y_test_one_hot.shape)

            eval_acc = model.evaluate(X_test, y_test_one_hot)
            print("eval_acc (loss, precision): ", eval_acc)

            # Prediction (Manual)
            raw_preds = model.predict(X_test)
            processed_preds = np.argmax(raw_preds, axis=1)

            # Print
            for i in range(len(X_test)):
                print("Ground Truth =", y_test[i], ", Prediction(has pedestrian)=",  processed_preds[i]==0, ", Raw Prediction =", raw_preds[i])
                plt.imshow(X_test_orig[i])
                plt.show()

if __name__ == "__main__":
	main()
