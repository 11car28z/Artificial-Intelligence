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
CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
print("2018250051 차수진")
print("\n\n################################################")
print("Exercise: train and fine-tune a Decision Tree for the moons dataset.")

print("\na. Generate a moons dataset using make_moons(n_samples=10000, noise=0.4).")
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

print("\nb. Split it into a training set and a test set using train_test_split().")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nc. Use grid search with cross-validation (with the help of the GridSearchCV class) \n"
"to find good hyperparameter values for a DecisionTreeClassifier. \n"
"Hint: try various values for max_leaf_nodes.")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)

print("\ngrid_search_cv.best_estimator_: \n", grid_search_cv.best_estimator_)
print("\nd. Train it on the full training set using these hyperparameters, and \n"
"measure your model's performance on the test set. You should get roughly 85% to 87% accuracy.")

from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)

print("\naccuracy_score(y_test, y_pred): \n", accuracy_score(y_test, y_pred))
