import data
import models
from sklearn.model_selection import train_test_split

#Define basic variables
num_classes = 36
inputShape = (32, 32, 1)

#Load images
[X, Y] = data.load_images(0, 10, num_classes)
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.20, shuffle= True)

#Get model
model = models.dummy1(inputShape, num_classes, x_train, y_train, x_valid, y_valid, True)

model = models.LeNet(inputShape, num_classes, x_train, y_train, x_valid, y_valid, True)