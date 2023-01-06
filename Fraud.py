# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 19:11:34 2022

@author: hughy
"""

import pandas as pd
import  numpy as np
import plotly.express as px
from plotly.offline import plot
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("fraud.csv")

#Dropping isFlaggedFraud since it has nothing to do with task
data = data.iloc[:,:-1]
#EDA
print(data.head())
print(data.isnull().sum()) #Checking for null values
print(data.type.value_counts())

type = data["type"].value_counts()
transactions = type.index
quantity = type.values

figure = px.pie(data, 
         values=quantity, 
         names=transactions,hole = 0.5, 
         title="Distribution of Transaction Type")
figure.show()
plot(figure)
    
#checking correlation between isFraud with other variables

#Transforming categorical features into numerical
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
print(data.head())

correlation = data.corr(numeric_only = True)
print(correlation["isFraud"].sort_values(ascending=False))    
    
#Based on the corrlation, drop features with below 0.01
#Additionally, Step is dropped since it only represent a unit of time
#Normalising the data
x = data[["type", "amount", "oldbalanceOrg"]]
y = data[["isFraud"]]

min_max_scaler = preprocessing.MinMaxScaler()
dummy1 = x.to_numpy() #returns a numpy array
scaled = min_max_scaler.fit_transform(dummy1)
x = pd.DataFrame(scaled)

#splitting data for validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77)

def find_Best_Classifier(x_train, x_test, y_train, y_test):
    rand=77
    names = ['LinearRegression','DecisionTree','NaiveBayes','RandomForest','MLP','GradientBossting','SVM']
    classifiers = [
                    Pipeline([('LinearRegression', linear_model.LinearRegression())
                              ]),
                    Pipeline([('DecisionTree', tree.DecisionTreeClassifier())
                              ]),
                    Pipeline([('NaiveBayes', GaussianNB())
                              ])
                   ]
    i=0
    for name, clf in zip(names, classifiers):
        
        clf.fit(x_train, y_train.values.ravel())
        score = clf.score(x_test, y_test)
        print(names[i],round(score*100,2))                 
        i+=1
    return 



#Since 
find_Best_Classifier(x_train, x_test, y_train, y_test)