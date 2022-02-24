import pandas as pd
import numpy as np
import os
#Formalization found here: https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf
#reading in training and prediction weights/labels
def format_data(test_file, prediction_file):
    df = pd.read_csv(test_file)
    df_pred = pd.read_csv(prediction_file)
    frames = [df['Weight'], df['Label'], df_pred['Pred_Label']]
    whole_frame = pd.concat(frames, axis=1)   #DF of sb labels and weights
    return whole_frame #return 3 column DF

#Calculating signal & background values
def sb_collection(whole_frame):
    signal = 0
    background = 0

    for i in range(len(whole_frame)):
        if whole_frame['Label'][i] == 's' and whole_frame['Pred_Label'][i] == 's': #signal vectors
            signal += float(whole_frame['Weight'][i])
        elif whole_frame['Label'][i] == 'b' and whole_frame['Pred_Label'][i] == 's': #background vectors
            background += float(whole_frame['Weight'][i])
        else:
            continue
    return (signal,background) #return tuple of signal and background

def perfect_sb_collection(whole_frame):
    signal = 0
    background = 0

    for i in range(len(whole_frame)):
        if whole_frame['Label'][i] == 's': #signal vectors
            signal += float(whole_frame['Weight'][i])
        else:
            continue
    return (signal,background) #return tuple of signal and background

#calculating AMS
def calc_AMS(sb,br=10):
    s = sb[0]
    b = sb[1]
    ams_squared = 2 * ((s + b + br) * np.log(1 + (s/(b+br)) ) - s)
    if ams_squared >= 0:
        ams = ams_squared**(1/2)
    elif ams_squared < 0: #check for complex sqrt
        print('The Radicand of the AMS is negative. Exiting...')
        exit()
    return(ams)

#completes full calculations
def run_AMS(test, pred):
    # train = input("Training Data File Name: ")
    # if os.path.exists(train) == True:
    #     pass
    # else:
    #     print(f'{train} not a valid file. Exiting...')
    #     exit()

    # pred = input("Label Prediction File Name: ")
    # if os.path.exists(pred) == True:
    #     pass
    # else:
    #     print(f'{pred} not a valid file. Exiting...')
    #     exit()
    # print(f'The calculated AMS value is: {calc_AMS(sb_collection(format_data(train,pred)),10)}')
    # train = 'Training_data/training_0_201.xlsx'
    # pred = 'pred_0_201.xlsx'
    # print(calc_AMS(sb_collection(format_data(train,pred)),10))
    return (calc_AMS(sb_collection(format_data(test,pred)),10), calc_AMS(perfect_sb_collection(format_data(test,pred)),10))

# print(run_AMS('Test_data/test_last_100.csv','pred_0_201.csv'))
