import data
import models
from sklearn.model_selection import train_test_split

#Define basic variables
num_classes = 36
inputShape = (32, 32, 3)
dataset = 'dataset1'

#Load images
# [X, Y] = data.load_images(dataset, 0, 20, num_classes)
# x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.20, shuffle= True)

#Get model
# model = models.mDummy1(inputShape, num_classes, x_train, y_train, x_valid, y_valid, 
#         steps_per_epoch = 40, epochs = 5, use_cache=False, dataset=dataset)

# model = models.mLeNet(inputShape, num_classes, x_train, y_train, x_valid, y_valid, 
#         steps_per_epoch = 20, epochs = 10, use_cache=True, dataset=dataset)
# [X, Y] = data.load_images(dataset, 50, 70, num_classes)
# models.evaluateModel(model, X, Y)

# model = models.mResNet50(inputShape, num_classes, x_train, y_train, x_valid, y_valid, False)



model = models.mResNet50(inputShape, num_classes,
        steps_per_epoch = 30, epochs = 40, use_cache=True, dataset=dataset)


# dataset = 'dataset2'
# model = models.mResNet50(inputShape, num_classes,
#         steps_per_epoch = 30, epochs = 40, use_cache=True, dataset=dataset)


