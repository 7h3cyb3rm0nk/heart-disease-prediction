import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# Impoting data preprocessing libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Importing model selection libraries.
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# Importing metrics for model evaluation.
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
# for knn
from sklearn.neighbors import KNeighborsClassifier
# Importing SMOTE for handling class imbalance.
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')


risk_df = pd.read_csv('data_cardiovascular_risk.csv', index_col='id')

numeric_features = []
categorical_features = []

# splitting features into numeric and categoric.

for col in risk_df.columns:  
  if risk_df[col].nunique() > 10:
    numeric_features.append(col) 
  else:
    categorical_features.append(col)

nan_columns = ['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate']

# dropping null values
risk_df.dropna(subset=nan_columns, inplace=True)

risk_df['glucose'] = risk_df.glucose.fillna(risk_df.glucose.median())

# we are going to replace the datapoints with upper and lower bound of all the outliers

def clip_outliers(risk_df):
    for col in risk_df[numeric_features]:
        # using IQR method to define range of upper and lower limit.
        q1 = risk_df[col].quantile(0.25)
        q3 = risk_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # replacing the outliers with upper and lower bound
        risk_df[col] = risk_df[col].clip(lower_bound, upper_bound)
    return risk_df

risk_df = clip_outliers(risk_df)

risk_df['sex'] = risk_df['sex'].map({'M':1, 'F':0})
risk_df['is_smoking'] = risk_df['is_smoking'].map({'YES':1, 'NO':0})

education_onehot = pd.get_dummies(risk_df['education'], prefix='education')

# drop the original education feature
risk_df.drop('education', axis=1, inplace=True)

# concatenate the one-hot encoded education feature with the rest of the data
risk_df = pd.concat([risk_df, education_onehot], axis=1)


# adding new column PulsePressure
risk_df['pulse_pressure'] = risk_df['sysBP'] - risk_df['diaBP']

# dropping the sysBP and diaBP columns
risk_df.drop(columns=['sysBP', 'diaBP'], inplace=True)

risk_df.drop('is_smoking', axis=1, inplace=True)


X = risk_df.drop('TenYearCHD', axis=1)
y= risk_df['TenYearCHD']


# from sklearn.ensemble import ExtraTreesClassifier

# model fitting
# model = ExtraTreesClassifier()
# model.fit(X,y)

# ranking feature based on importance
# ranked_features = pd.Series(model.feature_importances_,index=X.columns)
# model_df = risk_df.copy()

# X = model_df.drop(columns='TenYearCHD')     # independent features
# y = model_df['TenYearCHD']                  # dependent features


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2,stratify=y)

# smote = SMOTE(random_state=33)
# X_train, y_train = smote.fit_resample(X_train, y_train)
# print(X_train.shape)
# print(X_test.shape) 

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#models with highest recall score 
# knn=KNeighborsClassifier(n_neighbors=1)
# knn=KNeighborsClassifier(n_neighbors=2)
# knn=KNeighborsClassifier(n_neighbors=3)

#model with highest accuracy score
knn=KNeighborsClassifier(n_neighbors=22)

knn.fit(X_train,y_train)

# gridSearch for finding best k value

from sklearn.model_selection import GridSearchCV

# grid search for accuracy
param_grid = {'n_neighbors': np.arange(1, 101)} 
knn_test_accuracy = KNeighborsClassifier()
grid_accuracy = GridSearchCV(knn_test_accuracy, param_grid, cv=5, scoring='accuracy') 
grid_accuracy.fit(X_train, y_train)
print("Best K for accuracy:", grid_accuracy.best_params_['n_neighbors']) 
print("Best Score for accuracy:", grid_accuracy.best_score_)

#grid search for recall score
knn_test_recall = KNeighborsClassifier()
grid_recall = GridSearchCV(knn_test_recall, param_grid, cv=5, scoring='recall') 
grid_recall.fit(X_train, y_train)
print("Best K for recall:", grid_recall.best_params_['n_neighbors']) 
print("Best Score for recall:", grid_recall.best_score_)

import joblib

joblib.dump(knn , 'knnModel.joblib')
joblib.dump(scaler, 'scaler.joblib')
