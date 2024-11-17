#!/usr/bin/env python
# coding: utf-8

# import libraries

# In[12]:


import pandas as pd
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import seaborn as sns
#sns.set_style('whitegrid')
#matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
import sqlite3
import sqlite3
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import os


# In[13]:


import scipy


# In[14]:


from scipy import stats


# In[15]:


from scipy.stats import pearsonr


# In[16]:


from scipy.stats import ttest_ind


# In[17]:


from sklearn.metrics import classification_report


# In[18]:


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[19]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


# In[20]:


import requests


# In[21]:


import warnings
warnings.filterwarnings('ignore')


# In[22]:


# machine learning algorithm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score


# In[23]:


import warnings
warnings.filterwarnings('ignore')


# In[24]:


import math
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


# In[25]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

from sklearn.svm import SVC


# In[26]:


# evaluation on test data
from sklearn.metrics import classification_report,confusion_matrix


# In[27]:


pip install xlrd


# UPLOAD DATASET

# In[43]:


creditt = pd.read_csv(r"C:\Users\shedu\Downloads\clean_creditt.csv")


# In[45]:


from sklearn.metrics import classification_report,confusion_matrix


# In[47]:


pip install xlrd


# In[48]:


from sklearn.datasets import make_classification


# In[49]:


X, y = make_classification(n_samples=48000, n_features=19, n_informative=2, n_redundant=10,
                           n_clusters_per_class=1, weights=[0.8, 0.2], flip_y=0, random_state=42)


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #RANDOM FOREST- Optimisation using grid search

# In[51]:


X = creditt.drop(columns=['DEFAULT_PAYMENT'])
y =creditt['DEFAULT_PAYMENT']


# In[52]:


from sklearn.datasets import make_classification


# In[53]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}


# In[55]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)


# In[56]:


best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")


# In[57]:


from sklearn.model_selection import cross_val_score

rf_best = grid_search.best_estimator_
cv_scores = cross_val_score(rf_best, X_train, y_train, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean()}")


# In[60]:


y_pred = rf_best.predict(X_test)
y_train_pred=rf_best.predict(X_train)


# In[62]:


train_accuracy_rf_grid = round(accuracy_score(y_train_pred, y_train), 3)
accuracy_rf_grid = round(accuracy_score(y_pred, y_test), 3)
precision_score_rf_grid = round(precision_score(y_pred, y_test), 3)
recall_score_rf_grid = round(recall_score(y_pred, y_test), 3)
f1_score_rf_grid = round(f1_score(y_pred, y_test), 3)
auc_rf_grid = round(roc_auc_score(y_pred,y_test), 3)

print("The accuracy on train data is ", train_accuracy_rf_grid)
print("The accuracy on test data is ", accuracy_rf_grid)
print("The precision on test data is ", precision_score_rf_grid)
print("The recall on test data is ", recall_score_rf_grid)
print("The f1 on test data is ", f1_score_rf_grid)
print("The auc on test data is ", auc_rf_grid)

