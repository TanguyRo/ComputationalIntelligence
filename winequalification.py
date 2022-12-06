# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:32:38 2022

@author: Tanguy Robilliard, Killan Moal and Fréderic Forster
"""

# First of all, we have to import the required libraries necessary 
# Sklearn is necessary for using Artificial Intelligence
# and SVC seems to be the best classifier to use according to the data we study
import pandas as pd
import warnings

from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC #Might be the best option to consider
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, make_scorer, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout

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
#    ["log regression", LogisticRegression, log_reg_params],
#    ["decision tree", DecisionTreeClassifier, dec_tree_params],
    ["Random Forest", RandomForestClassifier, {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 400, 'random_state': 18}],
    ["K-Nearest Neighbors", KNeighborsClassifier, {'n_neighbors': 16}],
#    ["naive bayes", GaussianNB, naive_bayes_params],
    ["Support Vector Machines", SVC, {'C': 1.4, 'gamma': 1.3, 'kernel': 'rbf'}]
]

# In[Hyperparameter Tuning]

    # List of possible parameter for our SVC model
    paramSVC = {
        'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
        'kernel':['linear', 'rbf'],
        'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
        }
    
    # List of possible parameter for our Random Forest model
    paramRF = {
        'n_estimators': [200,300,400,500],
        'max_features': ['sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :["gini", "entropy"],
        'random_state' : [18]
        }
    
    # List of possible parameter for our K-Nearest Neighbors model
    k_range = list(range(1, 31))
    paramKNN = dict(n_neighbors=k_range)

# # Grid Search CV pour chaque classifier
# grid_SVC = GridSearchCV(SVC(), param_grid=paramSVC, scoring='accuracy', cv=10)
# grid_SVC.fit(X, y)
# best_param = grid_SVC.best_params_
# print("Support Vector Machine",best_param) #Support Vector Machine {'C': 1.4, 'gamma': 1.3, 'kernel': 'rbf'}
# grid_RF = GridSearchCV(RandomForestClassifier(), param_grid=paramRF, scoring='accuracy', cv=10)
# grid_RF.fit(X, y)
# print("Random Forest:",grid_RF.best_params_) #Random Forest: {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 400, 'random_state': 18}
# grid_KNN = GridSearchCV(KNeighborsClassifier(), param_grid=paramKNN, scoring='accuracy', cv=10)
# grid_KNN.fit(X, y)
# print("K-Nearest Neighbors:",grid_KNN.best_params_) #K-Nearest Neighbors: {'n_neighbors': 16}

# In[Cross Validation unbalanced]

# Creating the KFold
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

# In[Cross Validation balanced]

# # Creating the KFold
# crossValidation = KFold(n_splits=10, random_state=1, shuffle=True)

# insights = []
# for modelname, Model, params in modelclasses:
#         steps = [('over', RandomOverSampler()), ('model', Model(**params))]
#         pipeline = Pipeline(steps=steps)
#         scores = cross_val_score(pipeline, X, y, cv = crossValidation, scoring=make_scorer(classification_report_with_accuracy_score))
#         report = classification_report(originalclass, predictedclass)
#         insights.append((modelname, params, report))

# # insights.sort(key=lambda x:x[-1], reverse=True)
# for modelname, params, report in insights:
#     print(modelname, params)
#     print(report)

# In[No Cross validation unbalanced]

# Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# insights = []
# for modelname, Model, params_list in modelclasses:
#     for params in params_list:
#         model = Model(**params)        
#         model.fit(X_train, y_train)
#         Y_prediction = model.predict(X_test)
#         report = classification_report(y_test, Y_prediction)
#         insights.append((modelname, model, params, report))

# insights.sort(key=lambda x:x[-1], reverse=True)
# for modelname, model, params, report in insights:
#     print(modelname, params)
#     print(report)

# In[Artificial Neural Networks]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.34, random_state = 45)

# Initialize the constructor
model = Sequential()
 
# Add an input layer
model.add(Dense(16, activation ='relu', input_shape =(11, )))
 
# Add one hidden layer
model.add(Dense(14, activation ='relu'))

# Add one hidden layer
model.add(Dense(12, activation ='relu'))

# Add one hidden layer
model.add(Dense(10, activation ='relu'))

# Add one hidden layer
model.add(Dense(8, activation ='relu'))
 
# Add an output layer
model.add(Dense(1, activation ='relu'))

print(model.summary())
 
model.compile(loss ='binary_crossentropy', optimizer ='adam', metrics =['accuracy'])

# Training Model
model.fit(X_train, y_train, epochs = 100, batch_size = 32, verbose = 1)
  
# Predicting the Value
y_pred = model.predict(X_test)
print(y_pred)
