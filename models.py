from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
<<<<<<< HEAD
import tensorflow.keras.backend as K
=======
>>>>>>> 5e59f39660c0d939818a676ac7ccb5bcafb3a3fb
import helpers
import time
from modelsHelpers import saveModel, loadModel, existsModelCache, createAndSaveCurves, getTrainDatasetPath, getValidationDatasetPath


def mDummy1(input_shape, num_classes, steps_per_epoch, epochs, use_cache=False, dataset='dataset1'):
    model_name = 'dummy1'
    helpers.createFoldersForModel(model_name, dataset)
    print("===================== " + model_name + " model ====================")
    if existsModelCache(model_name, dataset) and use_cache:
        model = loadModel(model_name, dataset)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        evaluateModel(model, dataset, input_shape, steps_per_epoch, epochs)
        return model

    model = Sequential()
    # define the first (and only) CONV => RELU layer
    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=input_shape))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
<<<<<<< HEAD
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
=======
    model.compile(optimizer='adam',+
                  loss='categorical_crossentropy', metrics=['accuracy'])
>>>>>>> 5e59f39660c0d939818a676ac7ccb5bcafb3a3fb

    [history, time_taken] = fitModel(
        model, dataset, input_shape, steps_per_epoch, epochs)

    if use_cache:
        saveModel(model, model_name, dataset)

    createAndSaveCurves(history, model_name, dataset)

    return model


def mLeNet(input_shape, num_classes, steps_per_epoch, epochs, use_cache=False, dataset='dataset1'):
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

    model.add(Dense(num_classes))  # As classes are 36, model.add(Dense(36))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])

    # fitAndEvaluate(model, steps_per_epoch, epochs, x_train, y_train, x_valid, y_valid)

    [history, time_taken] = fitModel(
        model, dataset, input_shape, steps_per_epoch, epochs)

    if use_cache:
        saveModel(model, model_name, dataset)

    createAndSaveCurves(history, model_name, dataset)

    return history


def mResNet50(input_shape, num_classes, steps_per_epoch, epochs, use_cache=False, dataset='dataset1'):
    model_name = 'ResNet50'
    helpers.createFoldersForModel(model_name, dataset)
    print("===================== " + model_name + " model ====================")
    if existsModelCache(model_name, dataset) and use_cache:
        model = loadModel(model_name, dataset)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        evaluateModel(model, dataset, input_shape, steps_per_epoch, epochs)
        return model

    base_model = ResNet50(weights=None, include_top=False,
                          input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    [history, time_taken] = fitModel(
        model, dataset, input_shape, steps_per_epoch, epochs)

    print("Time taken, " + str(time_taken))
    if use_cache:
        saveModel(model, model_name, dataset)

    createAndSaveCurves(history, model_name, dataset)

    return model


def mResNet50_2(input_shape, num_classes, steps_per_epoch, epochs, use_cache=False, dataset='dataset1'):
    model_name = 'ResNet50_2'
    helpers.createFoldersForModel(model_name, dataset)
    print("===================== " + model_name + " model ====================")
    if existsModelCache(model_name, dataset) and use_cache:
        model = loadModel(model_name, dataset)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        evaluateModel(model, dataset, input_shape, steps_per_epoch, epochs)
        return model

    base_model = ResNet50(weights=None, include_top=False,
                          input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    [history, time_taken] = fitModel(
        model, dataset, input_shape, steps_per_epoch, epochs)

    print("Time taken, " + str(time_taken))
    if use_cache:
        saveModel(model, model_name, dataset)

    createAndSaveCurves(history, model_name, dataset)

    return model


def customVGG(input_shape, num_classes, steps_per_epoch, epochs, use_cache=False, dataset='dataset1'):
    model_name = 'Custom_VGG'
    helpers.createFoldersForModel(model_name, dataset)
    print("===================== " + model_name + " model ====================")
    # load model without classifier layers
    model = VGG16(include_top=False, input_shape=input_shape)
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(1024, activation='relu')(flat1)
    output = Dense(num_classes, activation='softmax')(class1)


    # define new model
    model = Model(inputs=model.inputs, outputs=[output, flat1])
    # summarize
    model.summary()


    n = 10 # number of features
<<<<<<< HEAD
    k = num_classes #number of classes
    cce = tf.keras.losses.CategoricalCrossentropy()
    def featuresLossFunction(y_true, y_pred):
    #toto treba ešte spraviť
        print("Shapes", y_true.shape, y_pred.shape)
        return K.mean(K.square(y_pred[:,0:n] - y_true[:,k:]))
=======
    k = 40 #number of classes
    cce = tf.keras.losses.CategoricalCrossentropy()
    def featuresLossFunction():
        def flf(y_true, y_pred):
        #toto treba ešte spraviť
            print(y_pred, y_true)
            return K.mean(K.square(y_pred[0:n] - y_true[k:]))
        return flf
>>>>>>> 5e59f39660c0d939818a676ac7ccb5bcafb3a3fb

    def customCCE(y_true, y_pred):
        return cce(y_true[0:k], y_pred).numpy()
        

    def customSPA(y_true,y_pred):
        return tf.keras.metrics.sparse_categorical_accuracy(y_true[0:k], y_pred)

    tf.executing_eagerly()


    model.compile(loss=[customCCE, featuresLossFunction],  metrics=[customSPA], optimizer='adam', run_eagerly=True)
    # model.run_eagerly = True

    [history, time_taken] = fitModel(
        model, dataset, input_shape, steps_per_epoch, epochs)

    print("Time taken, " + str(time_taken))
    if use_cache:
        saveModel(model, model_name, dataset)

    createAndSaveCurves(history, model_name, dataset)

    return model


def fitModel(model, dataset, input_shape, steps_per_epoch, epochs):
    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False)

    test_datagen = ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        getTrainDatasetPath(dataset),
        target_size=(input_shape[0], input_shape[1]),
        batch_size=steps_per_epoch,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        getValidationDatasetPath(dataset),
        target_size=(input_shape[0], input_shape[1]),
        batch_size=steps_per_epoch,
        class_mode='categorical')

    start = time.time()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples / steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / steps_per_epoch)

    return [history, time.time() - start]


def evaluateModel(model, dataset, input_shape, steps_per_epoch, epochs):
    test_datagen = ImageDataGenerator(
        featurewise_center=True, featurewise_std_normalization=True, rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        getValidationDatasetPath(dataset),
        target_size=(input_shape[0], input_shape[1]),
        batch_size=steps_per_epoch,
        class_mode='categorical')

    print("================= Evaluating the model =================")
    scores = model.evaluate_generator(validation_generator)
    print("%s: %.2f" % (model.metrics_names[0], scores[0]))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
