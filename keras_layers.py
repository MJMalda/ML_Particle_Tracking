import ams as ams
import csv
import data_formatter_csv as dafa
import numpy as np
import pandas as pd

training_file = 'Training_data/training.csv'
test_file = 'Test_data/test_last_100.csv'
prediction_file = 'pred_0_201.csv'

data_, label_ = dafa.inputs(training_file)
test_data,test_labels = dafa.inputs(test_file)

# ams_num = ams.run_AMS(training_file,prediction_file) #how to incorporate?
def fit_data(data,label):
     from keras.models import Sequential
     from keras.layers import Dense, Activation
     from keras.callbacks import EarlyStopping 

     model = Sequential()
     model.add(Dense(64,activation='relu',input_shape=(31,)))
     model.add(Dense(64,activation='relu',input_shape=(31,)))
     model.add(Dense(32,activation='relu',input_shape=(31,)))
     model.add(Dense(1,activation='sigmoid'))

     early_stopping_monitor = EarlyStopping(patience=5)

     model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
     model.fit(data,label,validation_split=0.3, epochs=100,callbacks=[early_stopping_monitor])
     return model

#predicts and formats prediction file
def make_predictions(test_data_,model):
     prediction = model.predict(test_data_)
     pred = []
     for i in range(len(prediction)):
          pred.append(str(int(round(prediction[i][0]))))

     #Writing to prediction file a list of 1's and 0's
     with open(prediction_file,'w') as f:
          pred_ = ['_'] + pred #to be replaced with header
          writer = csv.writer(f)
          writer.writerows(pred_)

     file = pd.read_csv(prediction_file)
     file.to_csv(prediction_file, header=['Pred_Label'], index=False)

     from sb_converter import outputs
     outputs(prediction_file)
     return pred

model_ = fit_data(data_,label_)
pred = make_predictions(test_data,model_)
ams_num = ams.run_AMS(test_file,prediction_file)
print(ams_num) #prints predicted ams vs perfect ams
# print(pred)
# print()
# print(test_labels)

#do we need to remove headers? Figure out if possible to input folder with colums out of order
#Incorporating AMS

#We have two different input shapes if we create a second dataset that omits all -999's. Then we can have two different models!