import os,random

# finalDir = 'out_final'
# if not os.path.exists(finalDir):
#     os.makedirs(finalDir)

# for filename in os.listdir('out'):
#     if not os.path.exists(finalDir + "/" + filename[0]):
#         os.makedirs(finalDir + "/" + filename[0])

#     os.rename('out/' + filename, finalDir + "/" + filename[0] + "/" + filename)

for letter in os.listdir('dataset3/train/'):
    if not os.path.exists('dataset3/validation/' + letter):
        os.makedirs('dataset3/validation/' + letter)

    for i in range(0,50):
        filetomove = random.choice(os.listdir('dataset3/train/' + letter))
        os.rename('dataset3/train/' + letter + '/' + filetomove, 'dataset3/validation/' + letter + '/' + filetomove)
    