from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow.keras.backend as K
import helpers
import time
from modelsHelpers import saveModel, loadModel, existsModelCache, createAndSaveCurves, getTrainDatasetPath, getValidationDatasetPath, createAndSaveCurvesFeatures
from customDataGenerator import CustomGenerator
from data import load_image_names_and_labels
import numpy as np


def customLeNet(input_shape, num_classes, steps_per_epoch, epochs, use_cache=False, dataset='dataset1'):
    model_name = 'customLeNet'
    helpers.createFoldersForModel(model_name, dataset)
    print("===================== " + model_name + " model ====================")
    if existsModelCache(model_name, dataset) and use_cache:
        model = loadModel(model_name, dataset)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        evaluateModel(model, dataset, input_shape, steps_per_epoch, epochs)
        return model

    model = Sequential()
    # First convolution Layer
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Second Convolution Layer

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    flat1 = Flatten()(model.layers[-1].output)

    dense1 = Dense(units=256, activation="relu")(flat1)
    classifierOutput = Dense(num_classes, activation='softmax', name='classifierOutput')(dense1)

    featuresOutput = Flatten(name='featuresOutput')(dense1)

    # define new model
    model = Model(inputs=model.inputs, outputs=[classifierOutput, featuresOutput])
    # summarize
    model.summary()

    n = 13 # number of features
    k = num_classes #number of classes
    cce = tf.keras.losses.CategoricalCrossentropy()
    def featuresLossFunction(y_true, y_pred):
        return K.square(y_pred[:,:n] - tf.cast(y_true, tf.float32)[:,k:])

    def classifierLossFunction(y_true, y_pred):
        return cce(y_true[:,0:k], y_pred)

    def classifierAccuracy(y_true,y_pred):
        return tf.keras.metrics.sparse_categorical_accuracy(tf.math.argmax(y_true[:,0:k],1), y_pred)

    def featuresAccuracy(y_true, y_pred):
        
        return K.sum(K.square(y_pred[:,:n] - tf.cast(y_true, tf.float32)[:,k:]))

    tf.executing_eagerly()

    losses = {'classifierOutput': classifierLossFunction, 'featuresOutput': featuresLossFunction}
    metrics = {'classifierOutput': classifierAccuracy, 'featuresOutput': featuresAccuracy}
    weights = {'classifierOutput':1.0, 'featuresOutput':1.0}

    model.compile(loss=losses,  metrics=metrics, optimizer='rmsprop', run_eagerly=True, loss_weights=weights)
    model.run_eagerly = True
    [history, time_taken] = fitModel(
        model, dataset, input_shape, steps_per_epoch, epochs, True)


    if use_cache:
        saveModel(model, model_name, dataset)

    createAndSaveCurvesFeatures(history, model_name, dataset)

    return model


def customLeNet2(input_shape, num_classes, steps_per_epoch, epochs, use_cache=False, dataset='dataset1'):
    model_name = 'customLeNet2'
    helpers.createFoldersForModel(model_name, dataset)
    print("===================== " + model_name + " model ====================")
    if existsModelCache(model_name, dataset) and use_cache:
        model = loadModel(model_name, dataset)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        evaluateModel(model, dataset, input_shape, steps_per_epoch, epochs)
        return model

    model = Sequential()
    # First convolution Layer
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Second Convolution Layer

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    flat1 = Flatten()(model.layers[-1].output)

    dense1 = Dense(units=256, activation="relu")(flat1)
    classifierOutput = Dense(num_classes, activation='softmax', name='classifierOutput')(dense1)

    featuresOutput = Flatten(name='featuresOutput')(model.layers[-1].output)

    # define new model
    model = Model(inputs=model.inputs, outputs=[classifierOutput, featuresOutput])
    # summarize
    model.summary()

    n = 13 # number of features
    k = num_classes #number of classes
    cce = tf.keras.losses.CategoricalCrossentropy()
    def featuresLossFunction(y_true, y_pred):
        return K.square(y_pred[:,:n] - tf.cast(y_true, tf.float32)[:,k:])

    def classifierLossFunction(y_true, y_pred):
        return cce(y_true[:,0:k], y_pred)

    def classifierAccuracy(y_true,y_pred):
        return tf.keras.metrics.sparse_categorical_accuracy(tf.math.argmax(y_true[:,0:k],1), y_pred)

    def featuresAccuracy(y_true, y_pred):
        
        return K.sum(K.square(y_pred[:,:n] - tf.cast(y_true, tf.float32)[:,k:]))

    tf.executing_eagerly()

    losses = {'classifierOutput': classifierLossFunction, 'featuresOutput': featuresLossFunction}
    metrics = {'classifierOutput': classifierAccuracy, 'featuresOutput': featuresAccuracy}
    weights = {'classifierOutput':1.0, 'featuresOutput':1.0}

    model.compile(loss=losses,  metrics=metrics, optimizer='rmsprop', run_eagerly=True, loss_weights=weights)
    model.run_eagerly = True
    [history, time_taken] = fitModel(
        model, dataset, input_shape, steps_per_epoch, epochs, True)


    if use_cache:
        saveModel(model, model_name, dataset)

    createAndSaveCurvesFeatures(history, model_name, dataset)

    return model


def mLeNetEnhanced(input_shape, num_classes, steps_per_epoch, epochs, use_cache=False, dataset='dataset1'):
    model_name = 'LeNet'
    helpers.createFoldersForModel(model_name, dataset)
    print("===================== " + model_name + " model ====================")
    if existsModelCache(model_name, dataset) and use_cache:
        model = loadModel(model_name, dataset)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        evaluateModel(model, dataset, input_shape, steps_per_epoch, epochs)
        return model

    model = Sequential()
    # First convolution Layer
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Second Convolution Layer

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])

    [history, time_taken] = fitModel(
        model, dataset, input_shape, steps_per_epoch, epochs, False)

    if use_cache:
        saveModel(model, model_name, dataset)

    createAndSaveCurves(history, model_name, dataset)

    return model


def mLeNetDefault(input_shape, num_classes, steps_per_epoch, epochs, use_cache=False, dataset='dataset1'):
    model_name = 'LeNetDefault'
    helpers.createFoldersForModel(model_name, dataset)
    print("===================== " + model_name + " model ====================")
    if existsModelCache(model_name, dataset) and use_cache:
        model = loadModel(model_name, dataset)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        evaluateModel(model, dataset, input_shape, steps_per_epoch, epochs)
        return model

    model = getDefaultModel(input_shape)

    model.add(Dense(num_classes, activation='softmax'))
    # model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop', metrics=['accuracy'])

    [history, time_taken] = fitModel(
        model, dataset, input_shape, steps_per_epoch, epochs, False)

    if use_cache:
        saveModel(model, model_name, dataset)

    createAndSaveCurves(history, model_name, dataset)

    return model

def mLeNetDefault_Features(input_shape, num_classes, steps_per_epoch, epochs, use_cache=False, dataset='dataset1'):
    model_name = 'LeNetDefault_Features'
    helpers.createFoldersForModel(model_name, dataset)
    print("===================== " + model_name + " model ====================")
    
    losses = {'classifierOutput': classifierLossFunction, 'featuresOutput': featuresLossFunction}
    metrics = {'classifierOutput': classifierAccuracy, 'featuresOutput': featuresAccuracy}
    weights = {'classifierOutput':1.0, 'featuresOutput':1.0}

    if existsModelCache(model_name, dataset) and use_cache:
        model = loadModel(model_name, dataset)
        model.compile(loss=losses,  metrics=metrics, optimizer='rmsprop', run_eagerly=True, loss_weights=weights)
        evaluateModel(model, dataset, input_shape, steps_per_epoch, epochs, True)
        return model

    model = getDefaultModel(input_shape)
    # flat1 = Flatten()(model.layers[-1].output)
    # dense1 = Dense(units=256, activation="relu")(flat1)
    classifierOutput = Dense(num_classes, activation='softmax', name='classifierOutput')(model.layers[-1].output)
    featuresOutput = Flatten(name='featuresOutput')(model.layers[-1].output)
    # define new model
    model = Model(inputs=model.inputs, outputs=[classifierOutput, featuresOutput])
    # summarize
    model.summary()

    model.compile(loss=losses,  metrics=metrics, optimizer='rmsprop', run_eagerly=True, loss_weights=weights)
    model.run_eagerly = True
    [history, time_taken] = fitModel(
        model, dataset, input_shape, steps_per_epoch, epochs, True)

    if use_cache:
        saveModel(model, model_name, dataset)

    createAndSaveCurvesFeatures(history, model_name, dataset)

    return model
    
def mLeNetDefault_Features_Dense(input_shape, num_classes, steps_per_epoch, epochs, use_cache=False, dataset='dataset1'):
    model_name = 'LeNetDefault_Features'
    helpers.createFoldersForModel(model_name, dataset)
    print("===================== " + model_name + " model ====================")
    
    losses = {'classifierOutput': classifierLossFunction, 'featuresOutput': featuresLossFunction}
    metrics = {'classifierOutput': classifierAccuracy, 'featuresOutput': featuresAccuracy}
    weights = {'classifierOutput':1.0, 'featuresOutput':1.0}

    if existsModelCache(model_name, dataset) and use_cache:
        model = loadModel(model_name, dataset)
        model.compile(loss=losses,  metrics=metrics, optimizer='rmsprop', run_eagerly=True, loss_weights=weights)
        evaluateModel(model, dataset, input_shape, steps_per_epoch, epochs, True)
        return model

    model = getDefaultModel(input_shape)
    # flat1 = Flatten()(model.layers[-1].output)
    dense1 = Dense(units=64, activation="relu")(model.layers[-1].output)
    classifierOutput = Dense(num_classes, activation='softmax', name='classifierOutput')(dense1)
    featuresOutput = Flatten(name='featuresOutput')(model.layers[-1].output)
    # define new model
    model = Model(inputs=model.inputs, outputs=[classifierOutput, featuresOutput])
    # summarize
    model.summary()

    model.compile(loss=losses,  metrics=metrics, optimizer='rmsprop', run_eagerly=True, loss_weights=weights)
    model.run_eagerly = True
    [history, time_taken] = fitModel(
        model, dataset, input_shape, steps_per_epoch, epochs, True)

    if use_cache:
        saveModel(model, model_name, dataset)

    createAndSaveCurvesFeatures(history, model_name, dataset)

    return model

def getDefaultModel(input_shape):
    model = Sequential()
    model.add(Conv2D(6, (5, 5), input_shape=input_shape))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, (5, 5)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    return model

def fitModel(model, dataset, input_shape, batch_size, epochs, addCharacteristics):
    [training_filenames, GT_training] = load_image_names_and_labels(dataset, 'train', addCharacteristics)
    [validation_filenames, GT_validation] = load_image_names_and_labels(dataset, 'validation', addCharacteristics)

    my_training_batch_generator = CustomGenerator(training_filenames, GT_training, batch_size)
    my_validation_batch_generator = CustomGenerator(validation_filenames, GT_validation, batch_size)

    start = time.time()
    history = model.fit(
        my_training_batch_generator,
        steps_per_epoch=len(training_filenames)/batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=my_validation_batch_generator,
        validation_steps=len(validation_filenames) / batch_size)

    return [history, time.time() - start]



def evaluateModel(model, dataset, input_shape, steps_per_epoch, epochs, addCharacteristics = False):
    # test_datagen = ImageDataGenerator(
    #     featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255)
    # validation_generator = test_datagen.flow_from_directory(
    #     getValidationDatasetPath(dataset),
    #     target_size=(input_shape[0], input_shape[1]),
    #     batch_size=steps_per_epoch,
    #     class_mode='categorical')
    [validation_filenames, GT_validation] = load_image_names_and_labels(dataset, 'validation', addCharacteristics)
    my_validation_batch_generator = CustomGenerator(validation_filenames, GT_validation, batch_size)

    print("================= Evaluating the model =================")
    scores = model.evaluate(my_validation_batch_generator)
    print("%s: %.2f" % (model.metrics_names[0], scores[0]))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
def featuresLossFunction(y_true, y_pred):
    k = 36 # number of classes
    n = 13 # number of features
    return K.square(y_pred[:,:n] - tf.cast(y_true, tf.float32)[:,k:])

def classifierLossFunction(y_true, y_pred):
    k = 36 # number of classes
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce(y_true[:,0:k], y_pred)

def classifierAccuracy(y_true,y_pred):
    k = 36 # number of classes
    return tf.keras.metrics.sparse_categorical_accuracy(tf.math.argmax(y_true[:,0:k],1), y_pred)

def featuresAccuracy(y_true, y_pred):
    k = 36 # number of classes
    n = 13 # number of features
    return K.sum(K.square(y_pred[:,:n] - tf.cast(y_true, tf.float32)[:,k:]))