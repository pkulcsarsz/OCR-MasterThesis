import data
import models
import modelsWithFeatures
import sys, getopt

# Define basic variables
num_classes = 36
inputShape = (32, 32, 3)
dataset = 'dataset2'

selectedModel = ""
print(selectedModel)



opts, args = getopt.getopt(sys.argv[1:], 'm:d:')
for opt, arg in opts:
    if opt == '-m':
        selectedModel = arg
    if opt == '-d':
        dataset = arg

if(dataset != 'dataset1' and dataset != 'dataset2'):
        print("Dataset is not recognized")
        raise NameError("Dataset is not recognized.")

if selectedModel == 'VGG':
  model = models.VGG(inputShape, num_classes, steps_per_epoch = 10, epochs = 40, use_cache=Falise, dataset=dataset)

if selectedModel == 'customVGG':
  model = modelsWithFeatures.customVGG(inputShape, num_classes, steps_per_epoch = 10, epochs = 10, use_cache=False, dataset=dataset)

