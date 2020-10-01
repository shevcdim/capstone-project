# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:34:17 2019

@author: shevcdim
"""
import math 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump, load
import pickle
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Set a random seed
import random
random.seed(42)

# Load the dataset
#in_file = 'C:\\Users\\shevcdim\\anaconda3\\PythonScripts\\normalized_incident_p.csv'
in_file = '2019oct-2020mar_inc.csv'

df = pd.read_csv(in_file,delimiter = ';',encoding = "ISO-8859-1")

#df = df.fillna(0)
display(df.columns)

df.drop(['cmdb_code', 'cmdb', 
       'location', 'u_incident.caller_id.location.u_country',
       'u_incident.caller_id.location.u_country.u_region',
       'u_incident.caller_id.u_segment'], axis=1, inplace=True)
display(df.columns)
categories = pd.get_dummies(df['Priority'], prefix='Priority')
display(data.columns)

#df.astype({'priority': 'int32'}).dtypes
#df['priority'] = df['priority'].map({5:0, 4:0, 3:1, 2:1, 1:1})
#data = pd.concat([df['description'], df['ldescription']], axis=1)
#data = pd.merge(df['description'], df['ldescription'], left_index=True, right_index=True)
#data = pd.get_dummies(df['cmdb_code'], prefix='cmdb')
#data = df.pop("description").fillna(df.pop("ldescription"teas )).astype(str)

#data = data.drop('rank', axis=1)
exit

df['descr'] = None
df['descr']=(df['description'] + ' ' + df['ldescription']).astype("str")
#df['descr'].replace(to_replace=r'^ba.$', value='new', regex=True)

df['descr'].replace(to_replace=r'\bError message', value='', regex=True, inplace=True)
df['descr'].replace(to_replace=r'\bHow it should work', value=' ', regex=True,inplace=True)
df['descr'].replace(to_replace=r'\bWhat is the Impact on business', value=' ', regex=True,inplace=True)
df['descr'].replace(to_replace=r'\bIs there any workaround available', value=' ', regex=True,inplace=True)
df['descr'].replace(to_replace=r'\bSelected Application', value=' ', regex=True,inplace=True)
df = df.fillna(0)




tokenizer = Tokenizer(num_words=1000)
l_2d = df['descr'].values.tolist()
#for i in range(500):
#    print(i, 'text', df['u_normalized_description'][i+184500])
tokenizer.fit_on_texts(l_2d)
cm= tokenizer.texts_to_sequences(l_2d)


X_train, X_test, y_train, y_test = train_test_split(cm, df['priority'], random_state=42, test_size = 0.4)

# One-hot encoding the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(len(X_train)))
print('Number of rows in the test set: {}'.format(len(X_test)))



#Tokenizer.fit_on_texts(self, texts)
'''
def get_one_hot_category(list_of_values,n):
    encoded_list = list([])
    for value in list_of_values:
        encoded_list.append(one_hot(value,n,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'))
    return np.array(encoded_list) 
'''
print (X_train[1])


# One-hot encoding the output into vector mode, each of length 1000
x_train = tokenizer.sequences_to_matrix(X_train, mode='count')
x_test = tokenizer.sequences_to_matrix(X_test, mode='count')
print(x_train[1])

print ('x train shape =', x_train.shape)
print ('x test shape =', x_test.shape)
print ('y shape =', y_train.shape)





model = Sequential()
model.add(Dense(512, activation='relu', input_dim=1000))
model.add(Dropout(0.5))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
#model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

model.compile(loss = 'mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

#model.fit(x_train, y_train, epochs=200, batch_size=100, verbose=1)
#model.fit(x_train, y_train, epochs=300,  verbose=1)
#model.fit(x_train, y_train,
#          batch_size=32,
#          epochs=20,
#          validation_data=(x_test, y_test), 
#          verbose=2)
model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          verbose=2)


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train)
print("\n Training Scores:", score[1])
score = model.evaluate(x_test, y_test)
print("\n Testing Scores:", score[1])

'''

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
y_pred1 = model.predict(x_test)
y_pred = np.argmax(y_pred1, axis=1)

# Print f1, precision, and recall scores
print('precision=',precision_score(y_test, y_pred , average="macro"))
print('recall=',recall_score(y_test, y_pred , average="macro"))
print('f1=',f1_score(y_test, y_pred , average="macro"))


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

# fit the model
history = model.fit(Xtrain, ytrain, validation_split=0.3, epochs=10, verbose=0)

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(Xtest, ytest, verbose=0)
'''
