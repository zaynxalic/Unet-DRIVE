import re

sng = "global correct: 73.9\naverage row correct: ['80.5', '29.0']\nIoU: ['72.9', '12.4']\nf1: 84.34607982635498\nmcc : 7.773429900407791\nrvd : -10.076527297496796\nmean IoU: 42.7"
print(re.findall(r'mcc : (.*)\n', sng)[0])