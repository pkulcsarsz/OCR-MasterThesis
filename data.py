import cv2
import numpy as np
import os


import keras
from keras.utils import plot_model

def load_images(ifrom, ito, num_classes, addGaus = False):
    letters = ["A", "B", "C", "D", "E" , "F", "G" , "H", "I", "J", "K", "L"
            , "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
            "X", "Y", "Z", "1" , "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    x_data = []
    y_data = []
    i = 0
    for letter in letters:
        j = 0
        for filename in os.listdir('data/' + letter):
            if j < ifrom or j > ito:
                j = j + 1
                continue
            sample = cv2.imread("data/" + letter + "/" + filename,cv2.IMREAD_GRAYSCALE)

            if addGaus == True:
                sample = addGausNoise(sample)

            sample = np.reshape(sample,(sample.shape[0],sample.shape[1],1))
            if sample is not None:                
                x_data.append(sample)
                y_data.append(i)
                j = j + 1
        i = i + 1

    x_data = np.array(x_data) / 255
    y_data = keras.utils.to_categorical(np.array(y_data),num_classes=num_classes)

    return[ x_data, y_data]

def addGausNoise(img, iterations = 1):
    result = img
    while iterations > 0 :
        gauss = np.random.normal(0,1,img.size)
        gauss = gauss.reshape(img.shape[0], img.shape[1], 1).astype('uint8')
        result = cv2.add(result, gauss)
        iterations = iterations - 1
    return result

def plotModel(model, fileName):
    plot_model(model, to_file=fileName)