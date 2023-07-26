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
print("Exercise: Grow a forest.")

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

print("\na. Continuing the previous exercise, generate 1,000 subsets of the training set, \n"
"each containing 100 instances selected randomly. \n"
"Hint: you can use Scikit-Learn's ShuffleSplit class for this.")

from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100

mini_sets = []

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

print("\n\nb. Train one Decision Tree on each subset, using the best hyperparameter values found above. \n"
"Evaluate these 1,000 Decision Trees on the test set. Since they were trained on smaller sets, \n"
"these Decision Trees will likely perform worse than the first Decision Tree, achieving only about 80% accuracy.")

from sklearn.base import clone

forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print("\nnp.mean(accuracy_scores): \n", np.mean(accuracy_scores))

print("\nc. Now comes the magic. For each test set instance, generate the predictions of the 1,000 Decision Trees, \n"
"and keep only the most frequent prediction (you can use SciPy's mode() function for this). \n"
"This gives you majority-vote predictions over the test set.")

Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

from scipy.stats import mode

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

print("\nd. Evaluate these predictions on the test set: you should obtain a slightly higher accuracy than \n"
"your first model (about 0.5 to 1.5% higher). Congratulations, you have trained a Random Forest classifier!")

print("\naccuracy_score(y_test, y_pred_majority_votes.reshape([-1])): \n", accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))