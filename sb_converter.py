import pandas as pd
import numpy as np
import csv
import os

def outputs(pred_file):
    df = pd.read_csv(pred_file)
    df = df.iloc[:]
    labels = df['Pred_Label'].to_numpy()
    label = []
    for i in range(len(labels)):
        if labels[i] == 0:
            label.append('s')
        elif labels[i] == 1:
            label.append('b')
    label = np.array(label)
    filler = np.array(['x'])
    with open(pred_file,'w') as f:
        label_ = np.append(filler,label)
        writer = csv.writer(f)
        writer.writerows(label_)

    file = pd.read_csv(pred_file)
    file.to_csv(pred_file, header=['Pred_Label'], index=False)

# file = 'pred_0_201.csv'
# print(outputs(file))