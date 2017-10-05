from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.initializers import RandomNormal, RandomUniform, he_normal, he_uniform, TruncatedNormal
from keras import utils
from random import *
import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image
import time


def main():
    # Read in data from .npy files
    training_labels = np.load('traininglabels.npy')
    training_images = np.load('trainingimages.npy')
    test_images = np.load('testingimages.npy')
    test_labels = np.load('testinglabels.npy')
    validation_images = np.load('validationimages.npy')
    validation_labels = np.load('validationlabels.npy')



    # Model Template

    model = Sequential()  # declare model
    model.add(Dense(1000, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
    model.add(Activation('tanh'))
    model.add(Dense(1000, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
    model.add(Activation('tanh'))
    model.add(Dense(1000, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
    model.add(Activation('tanh'))

    model.add(Dense(10, kernel_initializer='he_normal')) # last layer
    model.add(Activation('softmax'))


    # Compile Model
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    training_images = np.array(training_images)
    training_labels = np.array(training_labels)
    validation_labels = np.array(validation_labels)
    validation_images = np.array(validation_images)

    # Train Model
    history = model.fit(training_images, training_labels, validation_data = (validation_images, validation_labels), epochs=20, batch_size=1024)

    # Report Results

    # print(history.history)
    predictions = []
    for image in test_images:
        predictions.append(model.predict(image.reshape(1, 784)))

    # Save an image as test.png
    # for proof of concept of displaying images from array

    # im = Image.new("RGB", (28, 28))
    # pix = im.load()
    # for x in range(27):
    #     for y in range(27):
    #         pix[x, y] = (pictures[0][y][x], 0, 0)
    #
    # im.save("test.png", "PNG")

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


    # create the confusion matrix

    model.save('my_model.h5')


if __name__ == "__main__":
    main()
