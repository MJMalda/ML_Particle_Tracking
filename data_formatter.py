import pandas as pd
import numpy as np
import os

def inputs(training_file):
    df = pd.read_excel(training_file)
    df = df.iloc[:10000]
    labels = df['Label'].to_numpy()
    label = []
    for i in range(len(labels)):
        if labels[i] == 's':
            label.append(0)
        elif labels[i] == 'b':
            label.append(1)
    label = np.array(label)
    # data = df.drop(['Weight','Label'], axis=1)
    data=df.drop(['EventId','Label'], axis=1)
    jugular = []
    for i in range(len(data)):
        list = [data.iloc[i][j] for j in range(len(data.iloc[i]))]
        jugular.append(list)
    jugular = np.array(jugular)
    print(jugular)
    print(len(jugular))
    print(label)
    return jugular, label

inputs('Training_data/training_0_201.xlsx')