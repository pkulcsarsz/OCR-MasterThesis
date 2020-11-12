from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow.keras.backend as K
import helpers
import time
from modelsHelpers import saveModel, loadModel, existsModelCache, createAndSaveCurves, getTrainDatasetPath, getValidationDatasetPath
from dataGenerator import load_data_using_tfdata


def customVGG(input_shape, num_classes, steps_per_epoch, epochs, use_cache=False, dataset='dataset1'):
    model_name = 'Custom_VGG'
    helpers.createFoldersForModel(model_name, dataset)
    print("===================== " + model_name + " model ====================")
    # load model without classifier layers
    model = VGG16(include_top=False, input_shape=input_shape)
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)

    dense1 = Dense(units=4096, activation="relu")(flat1)
    dense2 = Dense(units=4096, activation="relu")(dense1)

    class1 = Dense(1024, activation='relu')(dense2)
    classifierOutput = Dense(num_classes, activation='softmax', name='classifierOutput')(class1)

    featuresOutput = Flatten(name='featuresOutput')(dense2)

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
    weights = {'classifierOutput':1.0, 'featuresOutput':10.0}

    model.compile(loss=losses,  metrics=metrics, optimizer='Adam', run_eagerly=True, loss_weights=weights)
    model.run_eagerly = True
    [history, time_taken] = fitModelTFLoad(
        model, dataset, input_shape, steps_per_epoch, epochs, True)


    print("Time taken, " + str(time_taken))
    if use_cache:
        saveModel(model, model_name, dataset)

    createAndSaveCurves(history, model_name, dataset)

    return model


def fitModelTFLoad(model, dataset, input_shape, steps_per_epoch, epochs, addCharacteristics):
    data_generator = load_data_using_tfdata(dataset, input_shape, steps_per_epoch,['train','validation'], addCharacteristics)

    start = time.time()
    history = model.fit(
        data_generator['train'],
        steps_per_epoch=data_generator['train_count'] / steps_per_epoch,
        epochs=epochs,
        validation_data=data_generator['validation'],
        validation_steps=data_generator['validation_count'] / steps_per_epoch)

    return [history, time.time() - start]

