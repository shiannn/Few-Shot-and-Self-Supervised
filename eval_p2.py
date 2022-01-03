import os
import pandas as pd

csv = pd.read_csv('test2.csv')
filenames = csv.loc[:,'filename']
labels = csv.loc[:,'label']
acc = 0
for filename, label in zip(filenames, labels):
    gt = os.path.splitext(filename)[0][:-5]
    acc += (gt == label)
print(acc, len(filenames))