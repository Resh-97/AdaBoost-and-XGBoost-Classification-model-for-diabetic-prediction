#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import libraries here
import numpy as np
# from sklearn import linear_model
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


# In[6]:


#Read datasets
diabetes_train = pd.read_csv("diabetes_train.csv")
diabetes_test = pd.read_csv("diabetes_test.csv")
X_train = diabetes_train.drop('Outcome',axis=1)
Y_train = diabetes_train['Outcome']
X_test = diabetes_test.drop('Outcome',axis=1)
Y_test = diabetes_test['Outcome']


# In[7]:


def train_eval(algorithm, grid_params, X_train, Y_train):
    reg_model = GridSearchCV(algorithm, grid_params, cv=5, n_jobs=-1, verbose=1)
    reg_model.fit(X_train, Y_train)
    parameters = reg_model.best_params_
    return parameters


# In[8]:


#Hyperparameter Tuning
AB_params ={'learning_rate' :[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
              'n_estimators':[50,100,150,200,250]}
params = train_eval(AdaBoostClassifier(), AB_params, X_train, Y_train)
print(params['learning_rate'])
print(params['n_estimators'])


# In[9]:


#AdaBoost Algorithm 
ab_model = AdaBoostClassifier(learning_rate=params['learning_rate'],n_estimators=params['n_estimators'])
ab_model. fit(X_train, Y_train)
y_pred = ab_model.predict(X_test)
cm1 = confusion_matrix(Y_test,y_pred)
Accuracy_adb = (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[1,0]+cm1[0,1]+cm1[1,1])
Sensitivity_adb = cm1[0,0]/(cm1[0,0]+cm1[0,1])
Specificity_adb = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print("Accuracy: \t", Accuracy_adb)
print("Sensitivity: \t",Sensitivity_adb )
print("Specificity: \t",Specificity_adb )


# In[10]:


#XGBoost Algorithm 
XGB_model = XGBClassifier(learning_rate=params['learning_rate'],n_estimators=params['n_estimators'])
XGB_model.fit(X_train, Y_train)
y_pred_xgb = ab_model.predict(X_test)
cm2 = confusion_matrix(Y_test,y_pred_xgb)
Accuracy_xgb = (cm2[0,0]+cm2[1,1])/(cm2[0,0]+cm2[1,0]+cm2[0,1]+cm2[1,1])
Sensitivity_xgb = cm2[0,0]/(cm2[0,0]+cm2[0,1])
Specificity_xgb = cm2[1,1]/(cm2[1,0]+cm2[1,1])
print("Accuracy: \t", Accuracy_xgb)
print("Sensitivity: \t",Sensitivity_xgb )
print("Specificity: \t",Specificity_xgb )


# In[11]:


if Accuracy_xgb > Accuracy_adb:
    Accuracy = Accuracy_xgb
else:
    Accuracy = Accuracy_adb
if Sensitivity_xgb > Sensitivity_adb:
    Sensitivity = Sensitivity_xgb
else:
    Sensitivity = Sensitivity_adb
if Specificity_xgb > Specificity_adb:
    Specificity = Specificity_xgb
else:
    Specificity = Specificity_adb


# In[ ]:


# Write output file
# Assuming iris_pred is DataFrame in the required output format
Output = [params['learning_rate'],params['n_estimators'],Accuracy.round(2),Sensitivity.round(2),Specificity.round(2)]
output = pd.DataFrame(Ouput)
output.to_csv('/output.csv', header=False, index=False)

