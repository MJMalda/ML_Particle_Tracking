import ams as ams
import csv
import data_formatter_csv as dafa
import numpy as np
import pandas as pd
from sb_converter import outputs

training_file = 'Training_data/training_0_201.csv'
test_file = 'Test_data/test_last_100.csv'
prediction_file = 'pred_0_201.csv'

data_0,data_1,data_23,label_0,label_1,label_23,weight_0,weight_1,weight_23 = dafa.inputs(training_file)
test_data_0,test_data_1,test_data_23,test_label_0,test_label_1,test_label_23,test_weight_0,test_weight_1,test_weight_23 = dafa.inputs(test_file)

# ams_num = ams.run_AMS(training_file,prediction_file) #how to incorporate?
def fit_data(data,label):
     from keras.models import Sequential
     from keras.layers import Dense, Activation
     from keras.callbacks import EarlyStopping 

     model = Sequential()
     model.add(Dense(30,activation='relu',input_shape=(np.shape(data)[-1],)))
     model.add(Dense(30,activation='relu',input_shape=(np.shape(data)[-1],)))
     model.add(Dense(20,activation='relu',input_shape=(np.shape(data)[-1],)))
     model.add(Dense(1,activation='sigmoid'))

     early_stopping_monitor = EarlyStopping(patience=5)

     model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
     model.fit(data,label,validation_split=0.2, epochs=100,callbacks=[early_stopping_monitor])
     return model

#predicts and formats prediction file
def make_predictions(test_data_,model):
     prediction = model.predict(test_data_)
     pred = []
     for i in range(len(prediction)):
          pred.append(str(int(round(prediction[i][0]))))
     return pred

model_0 = fit_data(data_0,label_0)
pred_0 = make_predictions(test_data_0,model_0)
model_1 = fit_data(data_1,label_1)
pred_1 = make_predictions(test_data_1,model_1)
model_23 = fit_data(data_23,label_23)
pred_23 = make_predictions(test_data_23,model_23)
pred_ = []
for i in pred_0:
     pred_.append(int(i))
for i in pred_1:
     pred_.append(int(i))
for i in pred_23:
     pred_.append(int(i))

#Writing to prediction file a list of 1's and 0's
pred_f = []
with open(prediction_file,'w') as f:
     pred_f = ['_'] + pred_ #to be replaced with header
     # print(pred_f)
     writer = csv.writer(f)
     writer.writerows(map(lambda x: [x],pred_f))

file = pd.read_csv(prediction_file)
file.to_csv(prediction_file, header=['Pred_Label'], index=False)
outputs(prediction_file)

#Rewriting test file
test_l = []
test_w = []
for i in test_label_0:
     test_l.append(int(i))
for i in test_label_1:
     test_l.append(int(i))
for i in test_label_23:
     test_l.append(int(i))

for i in test_weight_0:
     test_w.append(float(i))
for i in test_weight_1:
     test_w.append(float(i))
for i in test_weight_23:
     test_w.append(float(i))
d = {'Label':test_l,'Weight':test_w}
df = pd.DataFrame(d)
df.to_csv('generated_test.csv',index=False)
# with open('generated_test.csv','w') as f:
#      test_ = ['_'] + test_l #to be replaced with header
#      writer = csv.writer(f)
#      writer.writerows(map(lambda x: [x],df))

file = pd.read_csv('generated_test.csv')
file.to_csv('generated_test.csv', header=['Pred_Label','Weight'], index=False)
weight = pd.read_csv('generated_test.csv',usecols=['Weight'])
outputs('generated_test.csv')
#.to_csv('generated_test.csv', header=['Label','Weight'], index=False)

test = pd.read_csv('generated_test.csv',usecols=['Label'])
test = test['Label'].to_numpy()
print(test)
ams_num = ams.run_AMS(test_file,prediction_file)
print(f'Percentage of points predicted incorrectly: {sum(abs(np.array(pred_)-np.array(test)))/len(pred_)*100}%')
print(ams_num) #prints predicted ams vs perfect ams

#Do we need to remove headers? Figure out if possible to input folder with colums out of order
#Incorporating AMS -- Loss function? <-- Maybe some other loss model
#Seperate models for different jet models -- 3 or 4 different models
#Descending order in correlation to EventID so that we maintain correct order (Pandas?)
#Speed up data importing -- designate dtypes?
