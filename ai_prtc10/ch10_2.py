
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
CHAPTER_ID = "ensembles"
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
print("\nVoting Classifier")

print("\nExercise: Load the MNIST data and split it into a training set, a validation set, and a test set \n (e.g., use 50,000 instances for training, 10,000 for validation, and 10,000 fortesting).")

print("\nLoading the MNIST dataset.")

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.uint8)

from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)

print("\nExercise: Then train various classifiers, such as a Random Forest classifier, \n" "an Extra-Trees classifier, and an SVM.")

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(random_state=42)
mlp_clf = MLPClassifier(random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)

print("\n[estimator.score(X_val, y_val) for estimator in estimators]: \n", [estimator.score(X_val, y_val) for estimator in estimators])
print("\nThe linear SVM is far outperformed by the other classifiers. However, \n" "let's keep it for now since it may improve the voting classifier's performance.")
print("\nExercise: Next, try to combine them into an ensemble that outperforms them all \n" "on the validation set, using a soft or hard voting classifier.")

from sklearn.ensemble import VotingClassifier

named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]

voting_clf = VotingClassifier(named_estimators)

voting_clf.fit(X_train, y_train)

print("\nvoting_clf.score(X_val, y_val): \n", voting_clf.score(X_val, y_val))
print("\n[estimator.score(X_val, y_val) for estimator in voting_clf.estimators_]: ", [estimator.score(X_val, y_val) for estimator in voting_clf.estimators_])
print("\n\nLet's remove the SVM to see if performance improves. \n" "It is possible to remove an estimator by setting it to None using set_params() like this:")

print("\nvoting_clf.set_params(svm_clf=None): \n", voting_clf.set_params(svm_clf=None))

print("\nThis updated the list of estimators:")

print("\nvoting_clf.estimators: \n", voting_clf.estimators)

print("\nHowever, it did not update the list of trained estimators:")

print("\nvoting_clf.estimators_: \n", voting_clf.estimators_)

print("\nSo we can either fit the VotingClassifier again, \n" "or just remove the SVM from the list of trained estimators:")

del voting_clf.estimators_[2]

print("\nNow let's evaluate the VotingClassifier again:")

print("\nvoting_clf.score(X_val, y_val): \n", voting_clf.score(X_val, y_val))

print('\nA bit better! The SVM was hurting performance. \n'
"Now let's try using a soft voting classifier. \n"
'We do not actually need to retrain the classifier, we can just set voting to "soft":')

voting_clf.voting = "soft"

print("\nvoting_clf.score(X_val, y_val): \n", voting_clf.score(X_val, y_val))

print("\nNope, hard voting wins in this case.")

print("\nOnce you have found one, try it on the test set. \n"
"How much better does it perform compared to the individual classifiers?")

voting_clf.voting = "hard"
print("\nvoting_clf.score(X_test, y_test): \n", voting_clf.score(X_test, y_test))

print("\n\n[estimator.score(X_test, y_test) for estimator in voting_clf.estimators_]: \n",
[estimator.score(X_test, y_test) for estimator in voting_clf.estimators_])

print("\nThe voting classifier only very slightly reduced the error rate of the best model in this case.")