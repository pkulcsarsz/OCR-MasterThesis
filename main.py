import models
import sys, getopt
import helpers

# Define basic variables
num_classes = 36
inputShape = (32, 32, 3)
dataset = 'dataset3'
steps_per_epoch = 10
epochs = 10
useCache = False

selectedModel = ""
imagePathToTest = ""


opts, args = getopt.getopt(sys.argv[1:], 'm:d:e:s:c')
for opt, arg in opts:
  if opt == '-m':
    selectedModel = arg
  elif opt == '-d':
    dataset = arg
  elif opt == '-e':
    epochs = arg
  elif opt == '-s':
    steps_per_epoch = arg
  elif opt == '-c':
    if arg == "true":
      useCache = True
  elif opt == '-f':
    imagePathToTest = arg

if(dataset != 'dataset1' and dataset != 'dataset2' and dataset != 'dataset3'):
  print("Dataset is not recognized")
  raise NameError("Dataset is not recognized.")

if selectedModel == 'LeNetEnhancedFeaturesDense':
  model = models.mLeNetEnhanced_Features_Dense(inputShape, num_classes, steps_per_epoch = steps_per_epoch, epochs = epochs, use_cache=useCache, dataset=dataset)

if selectedModel == 'LeNetEnhancedFeatures':
  model = models.mLeNetEnhanced_Features(inputShape, num_classes, steps_per_epoch = steps_per_epoch, epochs = epochs, use_cache=useCache, dataset=dataset)

if selectedModel == 'LeNetEnhanced':
  model = models.mLeNetEnhanced(inputShape, num_classes, steps_per_epoch = steps_per_epoch, epochs = epochs, use_cache=useCache, dataset=dataset)

if selectedModel == 'LeNetDefault':
  model = models.mLeNetDefault(inputShape, num_classes, steps_per_epoch = steps_per_epoch, epochs = epochs, use_cache=useCache, dataset=dataset)

if selectedModel == 'LeNetDefaultFeatures':
  model = models.mLeNetDefault_Features(inputShape, num_classes, steps_per_epoch = steps_per_epoch, epochs = epochs, use_cache=useCache, dataset=dataset)


if selectedModel == 'LeNetDefaultFeaturesDense':
  model = models.mLeNetDefault_Features_Dense(inputShape, num_classes, steps_per_epoch = steps_per_epoch, epochs = epochs, use_cache=useCache, dataset=dataset)


if imagePathToTest != "":
  if(helpers.checkFileExists(fileToTest) == False):
    print("File to test is not found.")
    raise NameError("File to test is not found.")

  if(helpers.checkIfCacheExistsForModel(selectedModel, dataset) == False):
    print("Cache for the model not found. Please run training before using this command.")
    raise NameError("Cache for the model not found. Please run training before using this command.")

  imageToTest = loadImageFromPath(imagePathToTest)
  label = model.predict(imageToTest)
  print(label)

  #finish this







