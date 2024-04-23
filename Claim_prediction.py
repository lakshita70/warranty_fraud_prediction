# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 17:25:29 2021

@author: 91779
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

Warranty_claim = pd.read_csv("Warranty_Claim.csv")
#Eliminating Unwanted columns
Warranty_claim.drop(Warranty_claim.columns[[0]], axis = 1, inplace = True)

label_encoder = preprocessing.LabelEncoder()
Warranty_claim['Region']= label_encoder.fit_transform(Warranty_claim['Region'])
Warranty_claim['State']= label_encoder.fit_transform(Warranty_claim['State'])
Warranty_claim['Area']= label_encoder.fit_transform(Warranty_claim['Area'])
Warranty_claim['City']= label_encoder.fit_transform(Warranty_claim['City'])
Warranty_claim['Consumer_profile']= label_encoder.fit_transform(Warranty_claim['Consumer_profile'])
Warranty_claim['Product_category']= label_encoder.fit_transform(Warranty_claim['Product_category'])
Warranty_claim['Product_type']= label_encoder.fit_transform(Warranty_claim['Product_type'])
Warranty_claim['Purchased_from']= label_encoder.fit_transform(Warranty_claim['Purchased_from'])
Warranty_claim['Purpose']= label_encoder.fit_transform(Warranty_claim['Purpose'])

Warranty_claim.drop(Warranty_claim.columns[[0,1,2,3,]], axis = 1, inplace = True)
Warranty_claim.head()


# Declaring features & target
X = Warranty_claim.drop(['Fraud'], axis=1)
Y = Warranty_claim['Fraud']
## Using PCA instead of eleminating columns
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 25)
#X_new = pca.fit_transform(X)
#Resampling - SMOTE(Synthetic Minority Over Sampling Technique)
from imblearn.over_sampling import SMOTE
method = SMOTE(random_state = 7)
X_resampled, Y_resampled = method.fit_resample(X,Y)
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier

num_trees = 100
max_features = 5
kfold = KFold(n_splits=10, shuffle=False)
Random_Forest = RandomForestClassifier(n_estimators=num_trees, max_features=max_features, criterion="entropy",random_state=7)
results = cross_val_score(Random_Forest, X_resampled, Y_resampled, cv=kfold)
print(results.mean())
#Selected Random Forest
model = Random_Forest.fit(X,Y)
RandomForestClassifier=Random_Forest.predict(X)
from sklearn import metrics
print(metrics.classification_report(Y, RandomForestClassifier))
#Accuracy Score
from sklearn.metrics import accuracy_score

accuracy_score(Y,RandomForestClassifier)
import pickle
pickle.dump(model, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
