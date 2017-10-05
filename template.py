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

    # Model
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

    # cast data to arrays
    training_images = np.array(training_images)
    training_labels = np.array(training_labels)
    validation_labels = np.array(validation_labels)
    validation_images = np.array(validation_images)

    # Train Model
    history = model.fit(training_images, training_labels, validation_data = (validation_images, validation_labels), epochs=20, batch_size=1024)

    # Report Results
    print(history.history)

    # Create predictions from test data
    predictions = []
    for image in test_images:
        predictions.append(model.predict(image.reshape(1, 784)))

    # find what the prediction actually is
    prediction_labels_ints = []
    for p in predictions:
        # find the maximum value in the predictions list
        prediction_labels_ints.append(p.argmax())

    # make the one hot labels for the test data into ints
    test_labels_ints = []
    for l in test_labels:
        test_labels_ints.append(l.argmax())

    # create the confusion matrix
    confusion_matrix = np.empty((10, 10))
    confusion_matrix.fill(0)

    visualize_wrong = []
    test_wrong = []
    prediction_wrong = []

    for x in range(1624):
        # add one to the element in confusion_matrix
        confusion_matrix[test_labels_ints[x]][prediction_labels_ints[x]] = confusion_matrix[test_labels_ints[x]][prediction_labels_ints[x]] + 1
        # create a list of images that are incorrectly labelled
        if test_labels_ints[x]!=prediction_labels_ints[x] and len(visualize_wrong)<3:
            visualize_wrong.append(test_images[x].reshape(28, 28))
            test_wrong.append(test_labels_ints[x])
            prediction_wrong.append(prediction_labels_ints[x])

    print (confusion_matrix)

    # Save three images that were incorrectly identified
    im1 = Image.new("RGB", (28, 28))
    im2 = Image.new("RGB", (28, 28))
    im3 = Image.new("RGB", (28, 28))
    pix1 = im1.load()
    pix2 = im2.load()
    pix3 = im3.load()
    for x in range(27):
        for y in range(27):
            pix1[x, y] = (visualize_wrong[0][y][x], 0, 0)
            pix2[x, y] = (visualize_wrong[1][y][x], 0, 0)
            pix3[x, y] = (visualize_wrong[2][y][x], 0, 0)
    
    im1.save("visualization_image_1.png", "PNG")
    im2.save("visualization_image_2.png", "PNG")
    im3.save("visualization_image_3.png", "PNG")

    # print out the actual labels and predicted labels of images incorrectly identified
    print (test_wrong)
    print (prediction_wrong)

    # summarize history for accuracy
    # plot the model accuracy across epochs for trainng and validation data
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # save the model
    model.save('trained_model.h5')


if __name__ == "__main__":
    main()
