# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 04:31:44 2022

@author: Utilisateur_pret
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


bad = [0,1,2,3,4,5,6]
good = [7,8,9]
def ratingToClass(rating):
  if rating in good:
    return "Good"
  elif rating in bad:
    return "Bad"

def print_binary_evaluation(X_train, X_test,y_train, y_true, strategy):
    dummy_clf = DummyClassifier(strategy=strategy)
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict(X_test)
    results_dict = {'accuracy': accuracy_score(y_true, y_pred),
                    'recall': recall_score(y_true, y_pred, average = 'macro'),
                    'precision': precision_score(y_true, y_pred, average = 'macro'),
                    'f1_score': f1_score(y_true, y_pred, average = 'macro')}
    print(results_dict)
    return results_dict

wine = pd.read_csv('winequality-red.csv')

# Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1) # X is now like DF without quality column
y = wine['quality'] 

#count of the target variable
#sns.countplot(x='quality', data=wine)
wine['quality'] = wine['quality'].apply(ratingToClass)

#sns.countplot(x='quality', data=wine)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

print_binary_evaluation(X_train, X_test, y_train, y_test, 'most_frequent')
print_binary_evaluation(X_train, X_test,y_train, y_test, 'uniform')
print_binary_evaluation(X_train, X_test,y_train, y_test, 'stratified')
