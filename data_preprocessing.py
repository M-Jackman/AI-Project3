from keras import utils
from random import *
import numpy as np


zeros = []
ones = []
twos = []
threes = []
fours = []
fives = []
sixes = []
sevens = []
eights = []
nines = []
training_images = []
training_labels = []
validation_images = []
validation_labels = []

# Image Preprocessing

# get the images intp matrix of matrices
pictures = np.load('images.npy')
labels = np.load('labels.npy')


# turn all the matrices of images into vectors and add them to a list
flattened_matrices = []
for p in pictures:
    flattened_matrices.append(p.reshape(1, 784))

real_pics = []
for each in flattened_matrices:
    real_pics.append(each[0])

# turn all the matrices of labels into one-hot vectors
one_hot_vectors = []
for l in labels:
    one_hot_vectors.append(utils.to_categorical(l, 10))

real_labels = []
for each in one_hot_vectors:
    real_labels.append(each[0])

# separate data by number
for x in range(0, 6500):
    if labels[x] == 0:
        zeros.append(real_pics[x])
    elif labels[x] == 1:
        ones.append(real_pics[x])
    elif labels[x] == 2:
        twos.append(real_pics[x])
    elif labels[x] == 3:
        threes.append(real_pics[x])
    elif labels[x] == 4:
        fours.append(real_pics[x])
    elif labels[x] == 5:
        fives.append(real_pics[x])
    elif labels[x] == 6:
        sixes.append(real_pics[x])
    elif labels[x] == 7:
        sevens.append(real_pics[x])
    elif labels[x] == 8:
        eights.append(real_pics[x])
    elif labels[x] == 9:
        nines.append(real_pics[x])

# stratified sampling procedure

remaining_labels = real_labels
remaining_images = real_pics
num_takent = 0
for x in range(0, 3900):
    random_number = randint(0, 6499-num_takent)
    training_images.append(remaining_images[random_number])
    training_labels.append(remaining_labels[random_number])
    remaining_labels.pop(random_number)
    remaining_images.pop(random_number)
    num_takent += 1
    # training

num_takenv = 0
for x in range(0, 975):
    random_number = randint(0, 2599 - num_takenv)
    validation_images.append(remaining_images[random_number])
    validation_labels.append(remaining_labels[random_number])
    remaining_labels.pop(random_number)
    remaining_images.pop(random_number)
    num_takenv += 1
    # validation

test_labels = remaining_labels
test_images = remaining_images

# files for holding separated data
np.save("testinglabels", test_labels)
np.save("testingimages", test_images)
np.save("validationlabels", validation_labels)
np.save("validationimages", validation_images)
np.save("trainingimages", training_images)
np.save("traininglabels", training_labels)

