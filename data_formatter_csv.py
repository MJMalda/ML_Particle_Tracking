import pandas as pd
import numpy as np
import os
# import chardet
# print(chardet.detect(open('Training_data/training_0_201.csv', 'rb').read())['encoding'])

def inputs(training_file):
    df = pd.read_csv(training_file, dtype={'PRI_jet_num':'int8'},nrows=200)
    # df = df.iloc[:50000]
    # labels = df['Label'].to_numpy()
    # label = []
    # for i in range(len(labels)):
    #     if labels[i] == 's':
    #         label.append(0)
    #     elif labels[i] == 'b':
    #         label.append(1)
    # label = np.array(label)
    # data = df.drop(['Weight','Label'], axis=1)
    data=df.drop(['EventId'], axis=1)
    label_0 = data['Label'].loc[data['PRI_jet_num']==0]
    weight_0 = data['Weight'].loc[data['PRI_jet_num']==0]
    labels = label_0.to_numpy()
    _label_0 = []
    for i in range(len(labels)):
        if labels[i] == 's':
            _label_0.append(0)
        elif labels[i] == 'b':
            _label_0.append(1)
    label_0 = np.array(_label_0)
    label_1 = data['Label'].loc[data['PRI_jet_num']==1]
    weight_1 = data['Weight'].loc[data['PRI_jet_num']==1]
    labels = label_1.to_numpy()
    _label_1 = []
    for i in range(len(labels)):
        if labels[i] == 's':
            _label_1.append(0)
        elif labels[i] == 'b':
            _label_1.append(1)
    label_1 = np.array(_label_1)
    label_23 = data['Label'].loc[data['PRI_jet_num']>=2]
    weight_23 = data['Weight'].loc[data['PRI_jet_num']>=2]
    labels = label_23.to_numpy()
    _label_23 = []
    for i in range(len(labels)):
        if labels[i] == 's':
            _label_23.append(0)
        elif labels[i] == 'b':
            _label_23.append(1)
    label_23 = np.array(_label_23)
    data=data.drop(['Label','Weight'], axis=1)
    data_0 = (data.loc[data['PRI_jet_num']==0]).drop(['PRI_jet_leading_pt','PRI_jet_leading_eta','PRI_jet_leading_phi','PRI_jet_subleading_pt','PRI_jet_subleading_eta','PRI_jet_subleading_phi','DER_lep_eta_centrality','DER_prodeta_jet_jet','DER_mass_jet_jet','DER_deltaeta_jet_jet'], axis=1)
    data_1 = (data.loc[data['PRI_jet_num']==1]).drop(['PRI_jet_subleading_pt','PRI_jet_subleading_eta','PRI_jet_subleading_phi','DER_lep_eta_centrality','DER_prodeta_jet_jet','DER_mass_jet_jet','DER_deltaeta_jet_jet'], axis=1)
    data_23 = data.loc[data['PRI_jet_num']>=2]
    jet_0 = []
    jet_1 = []
    jet_23 = []
    for i in range(len(data_0)):
        list = [data_0.iloc[i][j] for j in range(len(data_0.iloc[i]))]
        jet_0.append(list)
    jet_0 = np.array(jet_0)
    for i in range(len(data_1)):
        list = [data_1.iloc[i][j] for j in range(len(data_1.iloc[i]))]
        jet_1.append(list)
    jet_1 = np.array(jet_1)
    for i in range(len(data_23)):
        list = [data_23.iloc[i][j] for j in range(len(data_23.iloc[i]))]
        jet_23.append(list)
    jet_23 = np.array(jet_23)
    # print(jugular)
    # print(len(jugular))
    # print(label)
    return jet_0, jet_1, jet_23, label_0, label_1, label_23, weight_0, weight_1, weight_23

# data_0,data_1,data_23,label_0,label_1,label_23=inputs('Training_data/training_0_201.csv')
# print(label_0)