from __future__ import division, print_function, unicode_literals
print("2018250051 차수진")
import numpy as np
import os

np.random.seed(42)

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training_linear_models"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

print("\n\nExercise: train an SVM regressor on the California housing dataset.\n")

print("\nLet's load the dataset using Scikit-Learn's fetch_california_housing() function:\n")

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]

print("\nSplit it into a training set and a test set:\n")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDon't forget to scale the data:\n")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nLet's train a simple LinearSVR first:\n")

from sklearn.svm import LinearSVR

lin_svr = LinearSVR(random_state=42)
lin_svr.fit(X_train_scaled, y_train)

print("\nLet's see how it performs on the training set:\n")

from sklearn.metrics import mean_squared_error

y_pred = lin_svr.predict(X_train_scaled)
mse = mean_squared_error(y_train, y_pred)
print("\nmse: ", mse)

print("\nLet's look at the RMSE:\n")
print("np.sqrt(mse): \n", np.sqrt(mse))

print("\nIn this training set, the targets are tens of thousands of dollars. \n"
"The RMSE gives a rough idea of the kind of error you should expect \n"
"(with a higher weight for large errors): so with this model we can expect \n"
"errors somewhere around $10,000. Not great. Let's see if we can do better \n"
"with an RBF Kernel. We will use randomized search with cross validation to \n"
"find the appropriate hyperparameter values for C and gamma:\n")

from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(SVR(), param_distributions, n_iter=10, verbose=2, cv=3, random_state=42)
rnd_search_cv.fit(X_train_scaled, y_train)

print("\nrnd_search_cv.best_estimator_: \n", rnd_search_cv.best_estimator_)

print("\nNow let's measure the RMSE on the training set:\n")

print("\nLooks much better than the linear model. \n"
"Let's select this model and evaluate it on the test set:\n")

y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

print("\nnp.sqrt(mse): \n", np.sqrt(mse))