from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout
from os import path

def dummy1(input_shape, num_classes, x_train, y_train, x_valid, y_valid, use_cache = False):
    model_name = 'dummy1'
    print("===================== " + model_name + " model ====================")
    if path.exists('cache/'+ model_name + '.h5') and path.exists('cache/'+ model_name + '.json') and use_cache :
        model = loadModel(model_name)
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

    model = fitAndEvaluate(model, 5, 10, x_train, y_train, x_valid, y_valid)

    if use_cache:
        saveModel(model, model_name)

    return model


def LeNet(input_shape, num_classes, x_train, y_train, x_valid, y_valid, use_cache = False):
    model_name = 'LeNet'
    print("===================== " + model_name + " model ====================")
    if path.exists('cache/'+ model_name + '.h5') and path.exists('cache/'+ model_name + '.json') and use_cache :
        model = loadModel(model_name)
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
    
    model.add(Flatten()) #Flatten the layers
    model.add(Dense(256)) # FC layer with 256 neurons
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes)) #As classes are 36, model.add(Dense(36))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # Use categorical_crossentropy as the number of classes are more than one.

    model = fitAndEvaluate(model, 5, 20, x_train, y_train, x_valid, y_valid)

    if use_cache:
        saveModel(model, model_name)

    return model

def fitAndEvaluate(model, steps_per_epoch, epochs, x_train, y_train, x_valid, y_valid):
    #Create history with fitting the data
    print(model.summary())
    fitModel(model, steps_per_epoch, epochs, x_train, y_train)
    evaluateModel(model, x_valid, y_valid)

    return model

def fitModel(model, steps_per_epoch, epochs, x_train, y_train):
    print("=================== Fitting the model ==================")
    history = model.fit(x_train ,y_train, steps_per_epoch = steps_per_epoch, epochs = epochs)


def evaluateModel(model, x_valid, y_valid):
    #Evaluate model
    print("================= Evaluating the model =================")
    scores = model.evaluate(x_valid, y_valid)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def saveModel(model, model_name):
    model_json = model.to_json()
    with open('cache/' + model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('cache/' + model_name + ".h5")
    print("Saved model to disk")

def loadModel(model_name):
    json_file = open('cache/' + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('cache/' + model_name + ".h5")
    print("Loaded model from disk")
    return loaded_model