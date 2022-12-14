# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:32:38 2022

@author: Tanguy Robilliard, Killan Moal and Fréderic Forster
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC #Might be the best option to consider
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, make_scorer, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import datasets, layers, models

bad = [0,1,2,3,4,5,6]
good = [7,8,9]

#warnings.filterwarnings("ignore")


def ratingToClass(rating):
  if rating in good:
    return 1
  elif rating in bad:
    return 0

# Variables for average classification report
originalclass = []
predictedclass = []
def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred) # return accuracy score (just because it's needed to return something)

# Loading the Data from the CSV dataset
wine = pd.read_csv('winequality-red.csv')
wine['quality'] = wine['quality'].apply(ratingToClass)

# Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1) # X is now like DF without quality column
y = wine['quality'] 

modelclasses = [
    ["decision tree", DecisionTreeClassifier, {'ccp_alpha': 0.01, 'criterion': 'entropy', 'max_depth': 11, 'max_features': 'sqrt'}],
    ["Random Forest", RandomForestClassifier, {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 400, 'random_state': 18}],
    ["K-Nearest Neighbors", KNeighborsClassifier, {'n_neighbors': 16}],
    ["naive bayes", GaussianNB, {'var_smoothing': 1.0}],
    ["Support Vector Machines", SVC, {'C': 1.4, 'gamma': 1.3, 'kernel': 'rbf'}]
]

# In[Hyperparameter Tuning]

# # List of possible parameter for our SVC model
# paramSVC = {
#     'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
#     'kernel':['linear', 'rbf'],
#     'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
#     }

# # List of possible parameter for our Random Forest model
# paramRF = {
#     'n_estimators': [200,300,400,500],
#     'max_features': ['sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :["gini", "entropy"],
#     'random_state' : [18]
#     }

# # List of possible parameter for our K-Nearest Neighbors model
# k_range = list(range(1, 31))
# paramKNN = dict(n_neighbors=k_range)

# # List of possible parameter for our Decision Tree model
# paramDT = {
#           'max_features': ['auto', 'sqrt', 'log2'],
#           'ccp_alpha': [0.1, .01, .001],
#           'max_depth': [4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],
#           'criterion': ['gini', 'entropy']
#           }

# # List of possible parameter for our Naive Bayes model
# paramNB = {'var_smoothing': np.logspace(0,-9, num=100)}

# # Grid Search CV pour SVC
# grid_SVC = GridSearchCV(SVC(), param_grid=paramSVC, scoring='accuracy', cv=10)
# grid_SVC.fit(X, y)
# best_param = grid_SVC.best_params_
# print("Support Vector Machine",best_param) #Support Vector Machine {'C': 1.4, 'gamma': 1.3, 'kernel': 'rbf'}

# # Grid Search CV pour Random Forest
# grid_RF = GridSearchCV(RandomForestClassifier(), param_grid=paramRF, scoring='accuracy', cv=10)
# grid_RF.fit(X, y)
# print("Random Forest:",grid_RF.best_params_) #Random Forest: {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 400, 'random_state': 18}

# # Grid Search CV pour K-Nearest Neighbour
# grid_KNN = GridSearchCV(KNeighborsClassifier(), param_grid=paramKNN, scoring='accuracy', cv=10)
# grid_KNN.fit(X, y)
# print("K-Nearest Neighbors:",grid_KNN.best_params_) #K-Nearest Neighbors: {'n_neighbors': 16}

# # Grid Search CV pour Decision Tree
# grid_SVC = GridSearchCV(DecisionTreeClassifier(), param_grid=paramDT, scoring='accuracy', cv=10)
# grid_SVC.fit(X, y)
# best_param = grid_SVC.best_params_
# print("Decision Tree",best_param)

# # Grid Search CV pour Naives Bayes
# grid_SVC = GridSearchCV(GaussianNB(), param_grid=paramNB, scoring='accuracy', cv=10)
# grid_SVC.fit(X, y)
# best_param = grid_SVC.best_params_
# print("Naive Baye",best_param)

# In[Cross Validation unbalanced]

# # Creating the KFold
# crossValidation = KFold(n_splits=10, random_state=1, shuffle=True)

# insights = []
# for modelname, Model, params in modelclasses:
#         model = Model(**params)
#         scores = cross_val_score(model, X, y, cv = crossValidation, scoring=make_scorer(classification_report_with_accuracy_score))
#         report = classification_report(originalclass, predictedclass)
#         insights.append((modelname, model, params, report))
 
# #insights.sort(key=lambda x:x[-1], reverse=True)
# for modelname, model, params, report in insights:
#     print(modelname, params)
#     print(report)

# In[Artificial Neural Networks]

# Split the dataset and balance dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
oversample = RandomOverSampler(sampling_strategy='minority')
X_train, y_train = oversample.fit_resample(X_train, y_train)

# Initialize the constructor
model = Sequential()
 
# Add an input layer
model.add(Dense(16, activation ='relu', input_shape =(11, )))
 
# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
 
model.compile(loss ='binary_crossentropy', optimizer ='adam', metrics =['accuracy'])

history = model.fit(X_train, y_train,epochs=30, batch_size=1, verbose=1)

y_pred = np.round(model.predict(X_test))
print(y_pred[0:10])

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy test: %.3f' % acc)
loss, acc = model.evaluate(X_train, y_train, verbose=0)
print('Test Accuracy train: %.3f' % acc)

print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=[0 ,1]), index=['true:Bad', 'true:Good'], columns=['pred:Bab', 'pred:Good']))

plt.subplot(211)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.legend()

# plt.subplot(211)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.legend()