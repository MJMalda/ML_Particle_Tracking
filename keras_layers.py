import ams as ams
import data_formatter_csv as dafa
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation

training_file = 'Training_data/training.csv'
test_file = 'Test_data/test_last_100.csv'
# prediction_file = 'pred_0_201.csv'

data, label = dafa.inputs(training_file)
# ams_num = ams.run_AMS(training_file,prediction_file) #how to incorporate?

model = Sequential()
model.add(Dense(32,activation='relu',input_shape=(31,)))
model.add(Dense(64,activation='relu',input_shape=(31,)))
model.add(Dense(64,activation='relu',input_shape=(31,)))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data,label,validation_split=0.1, epochs=15)

test_data,test_labels = dafa.inputs(test_file)
prediction = model.predict(test_data)
pred = []
for i in range(len(prediction)):
     pred.append(int(round(prediction[i][0])))
pred = np.array(pred)
print(sum(abs(test_labels - pred)))
print(pred,test_labels)
print(len(pred))