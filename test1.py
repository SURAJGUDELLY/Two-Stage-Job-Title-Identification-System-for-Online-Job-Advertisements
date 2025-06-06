from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import pickle
from nltk.stem import PorterStemmer
import os
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer #loading bert sentence model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial import distance
from numpy import dot
from numpy.linalg import norm
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint


bert_desc_X = np.load("model/bert_title_desc.npy")
Y = np.load("model/label.npy")
bert_title_X = np.load("model/bert_title.npy")

with open('model/desc_tfidf.txt', 'rb') as file:
    tfidf_desc_vector = pickle.load(file)
file.close()

with open('model/title_tfidf.txt', 'rb') as file:
    tfidf_title_vector = pickle.load(file)
file.close()

tfidf_title_X = np.load("model/tfidf_title_X.txt.npy")
tfidf_desc_X = np.load("model/tfidf_desc_X.txt.npy")

scaler1 = MinMaxScaler((0,1))
scaler2 = MinMaxScaler((0,1))
scaler3 = MinMaxScaler((0,1))
scaler4 = MinMaxScaler((0,1))

bert_desc_X = scaler1.fit_transform(bert_desc_X)
bert_title_X = scaler2.fit_transform(bert_title_X)
tfidf_desc_X = scaler3.fit_transform(tfidf_desc_X)
tfidf_title_X = scaler4.fit_transform(tfidf_title_X)

selected1 = SelectKBest(score_func = chi2, k = 300)
bert_desc_X = selected1.fit_transform(bert_desc_X, Y)

selected2 = SelectKBest(score_func = chi2, k = 300)
bert_title_X = selected2.fit_transform(bert_title_X, Y)

selected3 = SelectKBest(score_func = chi2, k = 300)
tfidf_desc_X = selected3.fit_transform(tfidf_desc_X, Y)

selected4 = SelectKBest(score_func = chi2, k = 300)
tfidf_title_X = selected4.fit_transform(tfidf_title_X, Y)

bert_desc_X_train, bert_desc_X_test, bert_desc_y_train, bert_desc_y_test = train_test_split(bert_desc_X, Y, test_size=0.2)
bert_title_X_train, bert_title_X_test, bert_title_y_train, bert_title_y_test = train_test_split(bert_title_X, Y, test_size=0.2)

tfidf_desc_X_train, tfidf_desc_X_test, tfidf_desc_y_train, tfidf_desc_y_test = train_test_split(tfidf_desc_X, Y, test_size=0.2)
tfidf_title_X_train, tfidf_title_X_test, tfidf_title_y_train, tfidf_title_y_test = train_test_split(tfidf_title_X, Y, test_size=0.2)

svm_cls = svm.SVC()
svm_cls.fit(tfidf_desc_X_train, tfidf_desc_y_train)
predict = svm_cls.predict(tfidf_desc_X_test)
acc = accuracy_score(tfidf_desc_y_test, predict)
print(acc)

nb_cls = GaussianNB()
nb_cls.fit(tfidf_desc_X_train, tfidf_desc_y_train)
predict = nb_cls.predict(tfidf_desc_X_test)
acc = accuracy_score(tfidf_desc_y_test, predict)
print(acc)

lr_cls = LogisticRegression()
lr_cls.fit(tfidf_desc_X_train, tfidf_desc_y_train)
predict = lr_cls.predict(tfidf_desc_X_test)
acc = accuracy_score(tfidf_desc_y_test, predict)
print(acc)

predict = []
for i in range(len(bert_desc_X_test)):
    max_value = 0
    pred = 0
    for j in range(len(bert_desc_X)):
        #dst = distance.euclidean(bert_desc_X_test[i], bert_desc_X_train[j])
        dst = dot(bert_desc_X_test[i], bert_desc_X[j]) / (norm(bert_desc_X_test[i]) * norm(bert_desc_X[j]))
        if dst > max_value and dst != 1:
            max_value = dst
            pred = Y[j]                       
    predict.append(pred)
acc = accuracy_score(bert_desc_y_test, predict)
print(acc)    

bert_desc_X_train = np.reshape(bert_desc_X_train, (bert_desc_X_train.shape[0], 10, 10, 3))
bert_desc_X_test = np.reshape(bert_desc_X_test, (bert_desc_X_test.shape[0], 10, 10, 3))

bert_desc_y_train = to_categorical(bert_desc_y_train)
bert_desc_y_test = to_categorical(bert_desc_y_test)

extension_model = Sequential()
extension_model.add(Convolution2D(32, (3 , 3), input_shape = (bert_desc_X_train.shape[1], bert_desc_X_train.shape[2], bert_desc_X_train.shape[3]), activation = 'relu'))
extension_model.add(MaxPooling2D(pool_size = (2, 2)))
extension_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
extension_model.add(MaxPooling2D(pool_size = (2, 2)))
extension_model.add(Flatten())
extension_model.add(Dense(units = 256, activation = 'relu'))
extension_model.add(Dense(units = bert_desc_y_train.shape[1], activation = 'softmax'))
extension_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = extension_model.fit(bert_desc_X_train, bert_desc_y_train, batch_size = 16, epochs = 50, validation_data=(bert_desc_X_test, bert_desc_y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    extension_model = load_model("model/cnn_weights.hdf5")
   
predict = extension_model.predict(bert_desc_X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(bert_desc_y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)


