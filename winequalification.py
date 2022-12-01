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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

bad = [0,1,2,3,4,5,6]
good = [7,8,9]
#warnings.filterwarnings("ignore")


def ratingToClass(rating):
  if rating in good:
    return 1
  elif rating in bad:
    return 0

# Loading the Data from the CSV dataset
wine = pd.read_csv('winequality-red.csv')
wine['quality'] = wine['quality'].apply(ratingToClass)

# Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1) # X is now like DF without quality column
y = wine['quality'] 

# Params for the different model
svc_params = [ {'kernel':'linear'}] #{"C":0.1}, {"C":0.8}, {"C":0.9}, {"C":1}, {"C":1.1}, {"C":1.2}, {"C":1.3}, {"C":1.4}, {'kernel':'rbf'}, 
kneighbors_params = [{"n_neighbors":3}, {"n_neighbors":5}]
rand_for_params = [{"criterion": "gini"}, {"criterion": "entropy"}]

modelclasses = [
#    ["log regression", LogisticRegression, log_reg_params],
#    ["decision tree", DecisionTreeClassifier, dec_tree_params],
#    ["random forest", RandomForestClassifier, rand_for_params],
#    ["k neighbors", KNeighborsClassifier, kneighbors_params],
#    ["naive bayes", GaussianNB, naive_bayes_params],
    ["support vector machines", SVC, svc_params]
]

# In[Cross Validation unbalanced]

# # Creating the KFold
# crossValidation = KFold(n_splits=10, random_state=1, shuffle=True)

# insights = []
# for modelname, Model, params_list in modelclasses:
#     for params in params_list:
#         model = Model(**params)
#         scores_accuracy = cross_val_score(model, X, y, scoring='accuracy', cv=crossValidation, n_jobs=-1)
#         scores_recall = cross_val_score(model, X, y, scoring='recall', cv=crossValidation, n_jobs=-1)
#         scores_f1 = cross_val_score(model, X, y, scoring='f1', cv=crossValidation, n_jobs=-1)
#         accuracy = round(mean(scores_accuracy),3)
#         recall = round(mean(scores_recall),3)
#         f1 = round(mean(scores_f1),3)
#         insights.append((modelname, model, params, accuracy, recall, f1))
 
# # insights.sort(key=lambda x:x[-1], reverse=True)
# for modelname, model, params, accuracy, recall, f1 in insights:
#     print(modelname, params)
#     print("mean accuracy score:",accuracy)
#     print("mean recall score:",recall)
#     print("mean f1 score:", f1)
#     print(" ")

# In[Cross Validation balanced]

# # Creating the KFold
crossValidation = KFold(n_splits=10, random_state=1, shuffle=True)

insights = []
for modelname, Model, params_list in modelclasses:
    for params in params_list:
        steps = [('over', RandomOverSampler()), ('model', Model(**params))]
        pipeline = Pipeline(steps=steps)
        model = Model(**params) # Only for the display
        scores_accuracy = cross_val_score(pipeline, X, y, scoring='accuracy', cv=crossValidation, n_jobs=-1)
        scores_recall = cross_val_score(pipeline, X, y, scoring='recall', cv=crossValidation, n_jobs=-1)
        scores_f1 = cross_val_score(pipeline, X, y, scoring='f1', cv=crossValidation, n_jobs=-1)
        accuracy = round(mean(scores_accuracy),3)
        recall = round(mean(scores_recall),3)
        f1 = round(mean(scores_f1),3)
        insights.append((modelname, model, params, accuracy, recall, f1))

# insights.sort(key=lambda x:x[-1], reverse=True)
for modelname, model, params, accuracy, recall, f1 in insights:
    print(modelname, params)
    print("mean accuracy score:",accuracy)
    print("mean recall score:",recall)
    print("mean f1 score:", f1)
    print(" ")

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

# tests de tanguy https://scikit-learn.org/stable/modules/cross_validation.html






























