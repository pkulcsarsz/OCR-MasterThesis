import os
from adversaries import generate_adversarial
from data import get_label_index, get_label
import numpy as np
import tensorflow as tf

letters = ["0", "1" , "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E" , "F", "G" , "H", "I", "J", "K", "L"
            , "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
            "X", "Y", "Z"]

def loadImageFromPath(path):
  image_raw = tf.io.read_file(path)
  image = tf.image.decode_image(image_raw)

  image = tf.cast(image, tf.float32)
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]

  return image

def createResultsForModels2(modelsArrays, modelsAreFeatures = [], dataset = 'dataset3'):
    total = 0
    amountOfModels = len(modelsArrays)
    cols = 1 + amountOfModels + amountOfModels^2
    results = np.zeros((37, cols))
    imagesToTest = [None]*(len(modelsArrays) + 1)

    for letter in letters:
        label_index = get_label_index(letters, letter)
        for filename in os.listdir(dataset + '/validation/' + letter):
            imagesToTest[0] = loadImageFromPath(dataset + '/validation/' + letter + '/' + filename)
            for i in range(len(modelsArrays)):
                imagesToTest[i + 1] = generate_adversarial(modelsArrays[i], imagesToTest[0], tf.one_hot(label_index, 36), 0.2, True)
            for i in range(len(modelsArrays)):
                #the basic image on each model
                if modelsAreFeatures[i] == True:
                    if get_label(letters, modelsArrays[i].predict(imagesToTest[0])[0])[0] == letter:
                        results[label_index, i*len(modelsArrays)] += 1
                        results[-1, i*len(modelsArrays)] += 1
                else:
                    if get_label(letters, modelsArrays[i].predict(imagesToTest[0]))[0] == letter:
                        results[label_index,i*len(modelsArrays)] += 1
                        results[-1, i*len(modelsArrays)] += 1
                #iterate trough adversarial samples for each model
                for j in range(len(modelsArrays)):
                    if modelsAreFeatures[i] == True:
                        if get_label(letters, modelsArrays[i].predict(imagesToTest[j + 1])[0])[0] == letter:
                            results[label_index, i*len(modelsArrays) + 1 + j] += 1
                            results[-1, i*len(modelsArrays) + 1 + j] += 1
                    else:
                        if get_label(letters, modelsArrays[i].predict(imagesToTest[j + 1]))[0] == letter:
                            results[label_index,i*len(modelsArrays) + 1 + j] += 1
                            results[-1, i*len(modelsArrays) + 1 + j] += 1

            #Increment the counter for each letter
            results[label_index,0] += 1
            total += 1
            if total % 250 == 0 :
                print("Current ===== ", total)

        #divide succes counts with total count of letter
        results[label_index, 1:] = results[label_index, 1:] / results[label_index, 0]

    results[-1, 1:] = results[-1, 1:] / results[-1, 0]
    print("Done")


def createResultsForModels(trained_model, trained_model_features):
    succ = 0
    succ_adv = 0
    succ_adv_f = 0
    succ_f = 0
    succ_f_adv = 0
    succ_f_adv_f = 0
    total = 0

    results = np.zeros((37,7))

    for letter in letters:
        label_index = get_label_index(letters, letter)
        for filename in os.listdir('dataset3/validation/' + letter):
            # Load the image
            imToTest = loadImageFromPath('dataset3/validation/' + letter + '/' + filename)
            # Generate adversarial sample 
            imToTest_adv_features = generate_adversarial(trained_model_features, imToTest, tf.one_hot(label_index, 36), 0.2, True)
            imToTest_adv = generate_adversarial(trained_model, imToTest, tf.one_hot(label_index, 36), 0.2, False)        
            # Get perdictions of the original and adversarial sample for basic model
            label = trained_model.predict(imToTest)
            label_adv = trained_model.predict(imToTest_adv)
            label_adv_f = trained_model.predict(imToTest_adv_features)
            # Get perdictions of the original and adversarial sample for features model
            label_f = trained_model_features.predict(imToTest)
            label_f_adv = trained_model_features.predict(imToTest_adv)
            label_f_adv_f = trained_model_features.predict(imToTest_adv_features)
            # Get label for classification from features labels
            label_f = label_f[0]
            label_f_adv = label_f_adv[0]
            label_f_adv_f = label_f_adv_f[0]
            #Check if the predictions were correct
            if (get_label(letters, label)[0] == letter):
                succ = succ + 1
                results[label_index,1] += 1
            if (get_label(letters, label_adv)[0] == letter):
                succ_adv = succ_adv + 1
                results[label_index,2] += 1
            if (get_label(letters, label_adv_f)[0] == letter):
                succ_adv_f = succ_adv_f + 1
                results[label_index,3] += 1
            
            if (get_label(letters, label_f)[0] == letter):
                succ_f = succ_f + 1
                results[label_index,4] += 1
            if (get_label(letters, label_f_adv)[0] == letter):
                succ_f_adv = succ_f_adv + 1
                results[label_index,5] += 1
            if (get_label(letters, label_f_adv_f)[0] == letter):
                succ_f_adv_f = succ_f_adv_f + 1
                results[label_index,6] += 1

            total += 1
            results[label_index,0] += 1

            if total % 250 == 0 :
                print("Current ===== ", total)
            
        results[label_index,1:] = results[label_index,1:] / results[label_index,0]

    results[36,0] = total
    results[36,1] = succ/total
    results[36,2] = succ_adv/total
    results[36,3] = succ_adv_f/total
    results[36,4] = succ_f/total
    results[36,5] = succ_f_adv/total
    results[36,6] = succ_f_adv_f/total

    print("FINISHED")
