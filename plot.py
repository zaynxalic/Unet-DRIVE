import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

"""
    Read the output result and generate the list of 
    1) epochs
    2) training_losses 
    3) learning_rates
    4) global corrects
"""

result = open('1105215644_unetpp_cbam_aspp_dropblock.txt' ,'r')
line = result.readline()
epochs = []
training_losses = []
learning_rates = []
global_corrects = []

dices = []
IoUs = []
f1s = []
mccs = []
IoU0s = []
IoU1s = []
mean_IoUs = []
precisions = []
recalls = []
rvds = []

epoch = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
epochs.append(epoch)

line = result.readline()
training_loss = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
training_losses.append(training_loss)

line = result.readline()
learning_rate = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
learning_rates.append(learning_rate)

line = result.readline()
dice = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
dices.append(dice)

line = result.readline()
global_correct = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
global_corrects.append(global_correct)

line = result.readline()
line = result.readline()
line = result.readline()
f1 = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[1])
f1s.append(f1)

line = result.readline()
try:
    mcc = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0]) 
except:
    mcc = 0
mccs.append(mcc)

line = result.readline()
try:
    rvd = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0]) 
except:
    rvd = 0
rvds.append(rvd)


line = result.readline()
mean_IoU =float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
mean_IoUs.append(mean_IoU)


while line:
    line = result.readline()
    if 'epoch' in line:
        
        epoch = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        epochs.append(epoch)

        line = result.readline()
        training_loss = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        training_losses.append(training_loss)

        line = result.readline()
        learning_rate = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        learning_rates.append(learning_rate)

        line = result.readline()
        dice = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        dices.append(dice)

        line = result.readline()
        global_correct = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        global_corrects.append(global_correct)

        line = result.readline()
        line = result.readline()
        line = result.readline()
        f1  = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[1])
        f1s.append(f1)

        line = result.readline()
        try:
            mcc = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0]) 
        except:
            mcc = 0
        mccs.append(mcc)

        line = result.readline()
        try:
            rvd = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0]) 
        except:
            rvd = 0
        rvds.append(rvd)


        line = result.readline()
        mean_IoU =float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        mean_IoUs.append(mean_IoU)
plot = {}
plot['training loss'] = training_losses
global_corrects = [float(i) for i in global_corrects]
plot['global corrects'] = np.asarray(global_corrects)/100
# plot['IoU0s'] = np.asarray(IoU0s)/100
# plot['IoU1s'] = np.asarray(IoU1s)/100
plot['mean iou'] = np.asarray(mean_IoUs)/100
max_arg = np.argmax(mean_IoUs)
print(max_arg)
print(f"Maximum IoU is {mean_IoUs[max_arg]}, ACC is {global_corrects[max_arg]}, DICE is {dices[max_arg]}, MCC is is {mccs[max_arg]}, RVD is {rvds[max_arg]}")

sns.lineplot(data=plot)

# print(epochs)
# print(training_losses)
# print(global_corrects)
# print(IoU0s)
# print(IoU1s)
# print(mean_IoUs)
