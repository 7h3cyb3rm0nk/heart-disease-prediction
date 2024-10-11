# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Importing data preprocessing libraries
from sklearn.preprocessing import StandardScaler

# Importing model selection libraries.
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# for knn
from sklearn.neighbors import KNeighborsClassifier

# Importing SMOTE for handling class imbalance.
from imblearn.over_sampling import SMOTE

"""### Importing the data"""

df = pd.read_csv("./data_cardiovascular_risk.csv", index_col="id")

"""### Data Cleaning"""

nan_columns = ["education", "cigsPerDay", "BPMeds", "totChol", "BMI", "heartRate"]

df.dropna(subset=nan_columns, inplace=True)

df["glucose"] = df["glucose"].fillna(df["glucose"].median())
df["sex"] = df["sex"].map({"M": 1, "F": 0})
df["is_smoking"] = df["is_smoking"].map({"YES": 1, "NO": 0})
df["pulse_pressure"] = df["sysBP"] - df["diaBP"]
df.drop(columns=["sysBP", "diaBP"], inplace=True)

"""#### Visually it seems like there is some outliers in totChol, glucose and pulse_pressure.

#### Using Modified-Z score to tackle the problem
"""

thresh = 3.5


# func to calcualte the modified z-score
def modified_z_score(series):
    median = np.median(series)
    mad = np.median(np.abs(series - median))

    return stats.norm.ppf(0.75) * (series - median) / mad


# iterate over all the columns and apply the modified z score function
cols_to_iterate = ["totChol", "glucose", "pulse_pressure", "heartRate"]

for col in cols_to_iterate:
    m_z_s = modified_z_score(df[col])
    df = df[m_z_s.abs() <= thresh]

X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]


"""### Handling Data Imbalance with SMOTE"""


from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy="minority")
X, y = smote.fit_resample(X, y)


"""### Split the data for training and testing"""

from sklearn.preprocessing import StandardScaler
import joblib


scaler = StandardScaler()
scaler.fit(X)

# save the scaler
joblib.dump(scaler, "scaler.joblib")


X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


"""### Finding the best parameters for KNN."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


clf = GridSearchCV(
    KNeighborsClassifier(),
    {
        "n_neighbors": [*range(1, 20, 4)],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    },
    cv=5,
    return_train_score=False,
)


clf.fit(X_train, y_train)
results = pd.DataFrame(clf.cv_results_)
results[
    ["param_algorithm", "param_n_neighbors", "param_weights", "mean_test_score"]
].sort_values(by=["mean_test_score"], ascending=False)


model = clf.best_estimator_

y_preds = model.predict(X_test)

joblib.dump(model, "model_KNeighborsClassifier.joblib")
