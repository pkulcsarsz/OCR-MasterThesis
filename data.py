import numpy as np
import os
from keras.utils.np_utils import to_categorical   

def get_label_index(letters, label):
    return np.argmax(np.asarray(letters) == label)

def load_image_names_and_labels(dataset = 'dataset3', phase = 'train', isFeature = False):
    letters = ["0", "1" , "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E" , "F", "G" , "H", "I", "J", "K", "L"
            , "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
            "X", "Y", "Z"]

    image_names = []
    image_labels = []

    characteristics = np.genfromtxt('characteristics2.csv', delimiter=',')

    path = dataset + '/' + phase + '/'
    for letter in os.listdir(path):
        #create label
        label_index = get_label_index(letters, letter)
        label = to_categorical(label_index, num_classes=36)
        if isFeature:
            label = np.concatenate((label, characteristics[label_index,:]))

        for sample in os.listdir(path + letter + '/'):
            image_labels.append(label)
            image_names.append(path + letter + '/' + sample)

    return [image_names, image_labels]

