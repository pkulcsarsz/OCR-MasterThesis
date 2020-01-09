from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from os import path
import helpers

def mDummy1(input_shape, num_classes, x_train, y_train, x_valid, y_valid, steps_per_epoch, epochs, use_cache = False, dataset = 'dataset1'):
    model_name = 'dummy1'
    print("===================== " + model_name + " model ====================")
    if existsModelCache(model_name, dataset) and use_cache :
        model = loadModel(model_name, dataset)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        evaluateModel(model, x_valid, y_valid)
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model = fitAndEvaluate(model, steps_per_epoch, epochs, x_train, y_train, x_valid, y_valid)

    if use_cache:
        saveModel(model, model_name, dataset)

    return model


def mLeNet(input_shape, num_classes, x_train, y_train, x_valid, y_valid, steps_per_epoch, epochs, use_cache = False, dataset = 'dataset1'):
    model_name = 'LeNet'
    print("===================== " + model_name + " model ====================")
    if existsModelCache(model_name, dataset) and use_cache :
        model = loadModel(model_name, dataset)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        evaluateModel(model, x_valid, y_valid)
        return model
        
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape)) # First convolution Layer
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #Second Convolution Layer
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten()) 
    model.add(Dense(256)) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes)) #As classes are 36, model.add(Dense(36))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model = fitAndEvaluate(model, steps_per_epoch, epochs, x_train, y_train, x_valid, y_valid)

    if use_cache:
        saveModel(model, model_name, dataset)

    return model

def mResNet50(input_shape, num_classes, x_train, y_train, x_valid, y_valid, steps_per_epoch, epochs, use_cache = False, dataset = 'dataset1'):
    model_name = 'ResNet50'
    print("===================== " + model_name + " model ====================")
    if existsModelCache(model_name, dataset) and use_cache :
        model = loadModel(model_name, dataset)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        evaluateModel(model, x_valid, y_valid)
        return model
    
    base_model = ResNet50(weights=None, include_top=False, input_shape= input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model = fitAndEvaluate(model, steps_per_epoch, epochs, x_train, y_train, x_valid, y_valid)

    if use_cache:
        saveModel(model, model_name, dataset)

    return model


def fitAndEvaluate(model, steps_per_epoch, epochs, x_train, y_train, x_valid, y_valid):
    #Create history with fitting the data
    print(model.summary())
    fitModel(model, steps_per_epoch, epochs, x_train, y_train)
    evaluateModel(model, x_valid, y_valid)

    return model

def fitModel(model, steps_per_epoch, epochs, x_train, y_train):
    print("=================== Fitting the model ==================")
    with helpers.Timer("Fitting"):
        history = model.fit(x_train ,y_train, steps_per_epoch = steps_per_epoch, epochs = epochs)


def evaluateModel(model, x_valid, y_valid):
    #Evaluate model
    print("================= Evaluating the model =================")
    scores = model.evaluate(x_valid, y_valid)
    print("%s: %.2f" % (model.metrics_names[0], scores[0]))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return scores


def saveModel(model, model_name, dataset):
    model_json = model.to_json()
    with open(getModelCachePath(model_name, dataset), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(getModelWeightsCachePath(model_name, dataset))
    print("Saved model to disk")

def loadModel(model_name, dataset):
    json_file = open(getModelCachePath(model_name, dataset), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(getModelWeightsCachePath(model_name, dataset))
    print("Loaded model from disk")
    return loaded_model

def getModelCachePath(model_name, dataset):
    return 'cache/' + dataset + '/' + model_name + '.json'

def getModelWeightsCachePath(model_name, dataset):
    return 'cache/' + dataset + '/' + model_name + '.h5'

def existsModelCache(model_name, dataset):
    return path.exists(getModelCachePath(model_name, dataset)) and path.exists(getModelWeightsCachePath(model_name, dataset))
