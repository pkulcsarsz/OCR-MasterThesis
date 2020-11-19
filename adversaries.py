import tensorflow as tf
import os, random
import matplotlib.pyplot as plt
import numpy as np

loss_object = tf.keras.losses.CategoricalCrossentropy()
loss_object = tf.keras.losses.MSE()

letters = ["0", "1" , "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E" , "F", "G" , "H", "I", "J", "K", "L"
            , "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
            "X", "Y", "Z"]

def create_adversarial_pattern(model, image, label):
    image = tf.cast(image, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = loss_object(label, prediction)
    
    gradient = tape.gradient(loss, image)
    
    signed_grad = tf.sign(gradient)
    
    return signed_grad

def generate_adversarial(model, image, true_label, epsilon = 0.1):
    return image + epsilon * create_adversarial_pattern(model, image, true_label)

def get_label(model_prediction):
  index = np.argmax(model_prediction)
  return [letters[index], index]

def get_label_index(label):
    return np.argmax(np.asarray(letters) == "3")

def loadRandomImage(letter = "0"):
    selected_image_path = random.choice(os.listdir("dataset2/validation/" + letter + "/"))

    image_raw = tf.io.read_file(selected_image_path)
    image = tf.image.decode_image(image_raw)

    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]

    return image

def display_images(image, model):
    label = model.predict(image)
    plt.figure()
    plt.imshow(image[0])
    plt.title(get_label(label))
    plt.show()

def testModelAdversary(model):
    test_label = "3"
    image = loadRandomImage(test_label)

    image_adv = generate_adversarial(model, image, get_label_index(test_label))

    display_images(image, model)
    display_images(image_adv, model)


