from skimage.io import imread
import numpy as np
import tensorflow as tf
import math

class CustomGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.image_filenames) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([imread(file_name) / 255 for file_name in batch_x]), np.array(batch_y)