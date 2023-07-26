print("2018250051 차수진")
import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import numpy as np
import os

np.random.seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

print("\n\nMNIST")

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
print("\nmnist.keys(): \n", mnist.keys())

X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)

print("\nX.shape: \n", X.shape)
print("\ny.shape: \n", y.shape)

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

print("\n\nExercise #01 An MNIST Classifier With Over 97% Accuracy")

from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)

print("\n\ngrid_search.best_params_: \n", grid_search.best_params_)

print("\ngrid_search.best_score_: \n", grid_search.best_score_)

from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(X_test)

print("\naccuracy_score(y_test, y_pred): \n", accuracy_score(y_test, y_pred))