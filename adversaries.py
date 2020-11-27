import tensorflow as tf
import os, random
import matplotlib.pyplot as plt
import numpy as np


letters = ["0", "1" , "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E" , "F", "G" , "H", "I", "J", "K", "L"
            , "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
            "X", "Y", "Z"]

def create_adversarial_pattern(model, image, label, isFeature):
    image = tf.cast(image, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        if isFeature:
            prediction = prediction[0]
            prediction = prediction[0]

        loss = tf.keras.losses.MSE(label, prediction[0])
        
    gradient = tape.gradient(loss, image)
    
    signed_grad = tf.sign(gradient)
    
    return signed_grad

def generate_adversarial(model, image, true_label, epsilon = 0.1, isFeature = False):
    return image + epsilon * create_adversarial_pattern(model, image, true_label, isFeature)