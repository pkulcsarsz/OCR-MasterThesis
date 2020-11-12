
from keras.models import model_from_json
from os import path
import matplotlib.pyplot as plt 

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

def getModelResultsPath(model_name, dataset):
    return 'results/' + model_name + "/" + dataset

def getTrainDatasetPath(dataset):
    return dataset + '/train'

def getValidationDatasetPath(dataset):
    return dataset + '/validation'

def createAndSaveCurves(history,model_name, dataset):
    createAndSaveLossCurve(history, model_name, dataset)
    createAndSaveAccCurve(history, model_name, dataset)

def createAndSaveLossCurve(history,model_name, dataset):
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.savefig(getModelResultsPath(model_name, dataset) + "/results_loss.png")

def createAndSaveAccCurve(history, model_name, dataset):
    plt.figure(figsize=[8,6])
    plt.plot(history.history['accuracy'],'r',linewidth=3.0)
    plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.savefig(getModelResultsPath(model_name, dataset) + "/results_acc.png")


  
def createAndSaveCurvesFeatures(history,model_name, dataset):
    #createAndSaveLossCurveFeatures(history, model_name, dataset)
    createAndSaveAccCurve(history, model_name, dataset)

def createAndSaveLossCurveFeatures(history,model_name, dataset):
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.savefig(getModelResultsPath(model_name, dataset) + "/results_loss.png")

def createAndSaveAccCurveFeatures(history, model_name, dataset):
    plt.figure(figsize=[8,6])
    plt.plot(history.history['classifierOutput_classifierAccuracy'],'r',linewidth=3.0)
    plt.plot(history.history['val_classifierOutput_classifierAccuracy'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.savefig(getModelResultsPath(model_name, dataset) + "/results_acc.png")