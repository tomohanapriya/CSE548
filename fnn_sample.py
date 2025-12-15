#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 14:19:05 2025

@author: ubuntu
"""

########################################
# Part 1 - Data Pre-Processing
#######################################

# To load a dataset file in Python, you can use Pandas. Import pandas using the line below
import pandas as pd
# Import numpy to perform operations on the dataset
import numpy as np

ScenarioA=['Training-a1-a3','Testing-a2-a4']
ScenarioB=['Training-a1-a2','Testing-a1']
ScenarioC=['Training-a1-a2','Testing-a1-a2-a3']
while 1:
    Scenario=input('Please enter the scenario you wish A OR B OR C:')
    if Scenario.lower() == 'a':
        TrainingData=ScenarioA[0]
        TestingData=ScenarioA[1]
        break
    elif Scenario.lower() == 'b':
        TrainingData=ScenarioB[0]
        TestingData=ScenarioB[1]
        break
    elif Scenario.lower() == 'c':
        TrainingData=ScenarioC[0]
        TestingData=ScenarioC[1]
        break
   


# Batch Size
BatchSize=10
# Epohe Size
NumEpoch=10

import data_preprocessor as dp
X_train,y_train = dp.get_processed_data(TrainingData+'.csv', './categoryMappings/',classType='binary')
X_test,y_test=dp.get_processed_data(TestingData+'.csv','./categoryMappings/',classType='binary')
print("Training data loaded:")
print("x train shape:",X_train.shape)
print("y train shape:",y_train.shape)
print("y train distribution:",np.bincount(y_train.astype(int)))
print("\ntesting data loaded:")
print("x test shape:",X_test.shape)
print("y test shape:",y_test.shape)
print("y test distribution:",np.bincount(y_test.astype(int)))


# Encoding categorical data (convert letters/words in numbers)
# Reference: https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
# The following code work without warning in Python 3.6 or older. Newer versions suggest to use ColumnTransformer
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [1, 2, 3])
X = onehotencoder.fit_transform(X).toarray()
'''
# The following code work Python 3.7 or newer
#rom sklearn.preprocessing import OneHotEncoder,
#om sklearn.compose import ColumnTransformer
#t = ColumnTransformer(
#   [('one_hot_encoder', OneHotEncoder(), [1,2,3])],    # The column numbers to be transformed ([1, 2, 3] represents three columns to be transferred)
#   remainder='passthrough'                         # Leave the rest of the columns untouched
#
# = np.array(ct.fit_transform(X), dtype=np.float)
#_train=sc.fit_transform(X_train)
#_test=sc.transform(X_test)
# Splitting the dataset into the Training set and Test set (75% of data are used for training)
# reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Perform feature scaling. For ANN you can use StandardScaler, for RNNs recommended is 
# MinMaxScaler. 
# referece: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://scikit-learn.org/stable/modules/preprocessing.html
#rom sklearn.preprocessing import StandardScaler
#c = StandardScaler()
#_train = sc.fit_transform(X_train)  # Scaling to the range [0,1]
#_test = sc.fit_transform(X_test)


########################################
# Part 2: Building FNN
#######################################

# Importing the Keras libraries and packages
#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#nput_dim=X_train.shape[1]
# Initialising the ANN
# Reference: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
classifier = Sequential()

# Adding the input layer and the first hidden layer, 6 nodes, input_dim specifies the number of variables
# rectified linear unit activation function relu, reference: https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=len(X_train[0])))


# Adding the second hidden layer, 6 nodes
#odel.add(Dense(units = 6, activation = 'relu'))
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
# Adding the output layer, 1 node, 
# sigmoid on the output layer is to ensure the network output is between 0 and 1
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print("\model compiled successfully")
# Compiling the ANN, 
# Gradient descent algorithm “adam“, Reference: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# This loss is for a binary classification problems and is defined in Keras as “binary_crossentropy“, Reference: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# Train the model so that it learns a good (or good enough) mapping of rows of input data to the output classification.
# add verbose=0 to turn off the progress report during the training
# To run the whole training dataset as one Batch, assign batch size: BatchSize=X_train.shape[0]
#istory =model.fit(X_train, y_train, batch_size = BatchSize, epochs = NumEpoch)
classifierHistory = classifier.fit(X_train,y_train,batch_size=BatchSize,epochs=NumEpoch)
# evaluate the keras model for the provided model and dataset
loss, accuracy =classifier.evaluate(X_train,y_train)
print('Print the loss and the accuracy of the model on the dataset')
print('Loss [0,1]: %.4f' % (loss), 'Accuracy [0,1]: %.4f' % (accuracy))

########################################
# Part 3 - Making predictions and evaluating the model
#######################################

# Predicting the Test set result
y_pred=classifier.predict(X_test,verbose=0)
print("\n max predictions:",y_pred.max())
print("\nMean predictions:",y_pred.mean())
print("predictions>0.5:",(y_pred>0.5).sum())
print("predictions>0.9",(y_pred>0.9).sum())
y_pred_binary = (y_pred > 0.5).astype(int).flatten()
print("\npredicted class distribution:")
print("class 0 (normal):",(y_pred_binary == 0).sum())
print("class 1 (attack):",(y_pred_binary==1).sum())
   # y_pred is 0 if less than 0.9 or equal to 0.9, y_pred is 1 if it is greater than 0.9
# summarize the first 5 cases
#for i in range(5):
#	print('%s => %d (expected %d)' % (X_test[i].tolist(), y_pred[i], y_test[i]))

# Making the Confusion Matrix
# [TN, FP ]
# [FN, TP ]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_binary)
print('Print the Confusion Matrix:')
print('[ TN, FP ]')
print('[ FN, TP ]=')
print(cm)

if cm.shape==(1,1):
    print("\model")
    if y_pred_binary[0]==0:
        TN = CM[0,0]
        FP = 0
        FN = 0
        TP = 0
    else:
        TN = 0
        FP = 0
        FN = 0
        TP = cm[0,0]
elif cm.shape == (2,2):
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TP = cm[1,1]
else:
    print("\nerror,cm.shape",cm.shape)
    TN = FP = FN = TP = 0
    
total = TN+FP+FN+TP
if total>0:
    testing_accuracy = (TP+TN)/total
else:
    testing_accuracy = 0.0
    

        


########################################
# Part 4 - Visualizing
#######################################

# Import matplot lib libraries for plotting the figures. 
import matplotlib.pyplot as plt

# You can plot the accuracy
print('Plot the accuracy')
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
# Keras 2.2.4 recognizes 'acc' and 2.3.1 recognizes 'accuracy'
# use the command python -c 'import keras; print(keras.__version__)' on MAC or Linux to check Keras' version
plt.plot(classifierHistory.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('accuracy_sample.png')
plt.grid(True)
plt.show()

# You can plot history for loss
print('Plot the loss')
plt.subplot(1,2,2)
plt.plot(classifierHistory.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.grid(True)
plt.savefig('loss_sample.png')
plt.show()

plt.tight_layout()
plt.savefig('result'+Scenario.lower()+'.png',dpi=150)
print('saved plot as:result'+Scenario.lower()+'.png')
plt.show()

print("complete!check your results above.")
print("="*50)
