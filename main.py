import data
import models
import sys, getopt

# Define basic variables
num_classes = 36
inputShape = (32, 32, 3)
dataset = 'dataset3'
steps_per_epoch = 10
epochs = 10

selectedModel = ""

opts, args = getopt.getopt(sys.argv[1:], 'm:d:e:s')
for opt, arg in opts:
  if opt == '-m':
    selectedModel = arg
  if opt == '-d':
    dataset = arg
  if opt == '-e':
    epochs = arg
  if opt == '-s':
    steps_per_epoch = arg

if(dataset != 'dataset1' and dataset != 'dataset2' and dataset != 'dataset3'):
  print("Dataset is not recognized")
  raise NameError("Dataset is not recognized.")

if selectedModel == 'customLeNet':
  model = models.customLeNet(inputShape, num_classes, steps_per_epoch = steps_per_epoch, epochs = epochs, use_cache=False, dataset=dataset)

if selectedModel == 'mLeNet':
  model = models.mLeNet2(inputShape, num_classes, steps_per_epoch = steps_per_epoch, epochs = epochs, use_cache=False, dataset=dataset)









