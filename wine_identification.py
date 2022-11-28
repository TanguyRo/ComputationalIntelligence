# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:32:38 2022

@author: tanguy
"""

# First of all, we have to import the resuired libraries necessary 
# Sklearn is necessary for using Artificial Intelligence
# and SVC seems to be the best classifier to use according to the data we study

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC #Might be the best option to consider
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import seaborn as sns
import matplotlib.pyplot as plt

# Loading the Data from Kaggle CSV file to DataFrame format from Pandas Library
df = pd.read_csv('winequality-red.csv')

# In[Barplot]
"""
Getting Barplots from Seaborn Library for :
     - fixed acidity
     - volatile acidity
     - citric acid
     - residual sugar
     - chloride
     - free sulfur dioxyde
     - total sulfur dioxyde
     - sulphates
     - alcohol
"""
#Here we see that fixed acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = df)

#Here we see that its quite a downing trend in the volatile acidity as we go higher the quality 
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = df)

#Composition of citric acid go higher as we go higher in the quality of the df
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = df)

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = df)


#Composition of chloride also go down as we go higher in the quality of the df
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = df)


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = df)

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = df)

#Sulphates level goes higher with the quality of df
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = df)

#Alcohol level also goes higher as te quality of df increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = df)

# In[Classification]

#Making binary classificaion for the response variable.
#Dividing df as good and bad by giving the limit for the quality

# Using cut to segment and sort wine values into bins. 
# Useful for going from a continuous variable to a categorical variable.

# Setting values for the bins to sort it in 2 groups, Good wine and Bad wine
bins = (2, 6.5, 8) 
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)

# Using LabelEncoder Transformer from SKLearn Library to assign labels to quality variable
label_quality = LabelEncoder()

#Bad becomes 0 and good becomes 1 
df['quality'] = label_quality.fit_transform(df['quality'])

# df['quality'].value_counts()

sns.countplot(df['quality'])

#Now seperate the dataset as response variable and feature variabes
X = df.drop('quality', axis = 1) # X is now like DF without quality column
y = df['quality'] 

#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Applying Standard scaling to get optimized result
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# In[RandomForestClassifier]
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

# checking how the model performed
print(classification_report(y_test, pred_rfc))

#Confusion matrix for the random forest classification
print(confusion_matrix(y_test, pred_rfc))

# In[SGDClassifier]
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)

# checking how the model performed
print(classification_report(y_test, pred_sgd))

# Confusion matrix for SGD classification
print(confusion_matrix(y_test, pred_sgd))


# In[SVC]
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

# checking how the model performed
print(classification_report(y_test, pred_svc))


#Finding best parameters for our SVC model
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}

grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)

grid_svc.fit(X_train, y_train)

#Best parameters for our svc model
grid_svc.best_params_

#Let's run our SVC again with the best parameters.
svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))

#Now lets try to do some evaluation for random forest model using cross validation.
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
rfc_eval.mean()







































