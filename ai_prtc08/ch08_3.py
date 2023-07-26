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

print("\n\nExercise: train an SVM classifier on the MNIST dataset. \n"
"Since SVM classifiers are binary classifiers, you will need to use \n"
"one-versus-all to classify all 10 digits. You may want to tune \n"
"the hyperparameters using small validation sets to speed up the process. \n"
"What accuracy can you reach?\n")

print("\nFirst, let's load the dataset and split it into a training set and \n"
"a test set. We could use train_test_split() but people usually just take \n"
"the first 60,000 instances for the training set, and the last 10,000 instances \n"
"for the test set (this makes it possible to compare your model's performance with others):\n")

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)

X = mnist["data"]
y = mnist["target"].astype(np.uint8)

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

print("\nMany training algorithms are sensitive to the order of the training instances, \n"
"so it's generally good practice to shuffle them first. However, \n"
"the dataset is already shuffled, so we do not need to do it.\n")

print("\nLet's start simple, with a linear SVM classifier. \n"
"It will automatically use the One-vs-All (also called One-vs-the-Rest, OvR) strategy,\n"
"so there's nothing special we need to do. Easy!\n")

lin_clf = LinearSVC(random_state=42)
print("\nlin_clf.fit(X_train, y_train): \n", lin_clf.fit(X_train, y_train))

print("\nLet's make predictions on the training set and measure the accuracy \n"
"(we don't want to measure it on the test set yet, since we have not selected \n"
"and trained the final model yet):\n")

from sklearn.metrics import accuracy_score

y_pred = lin_clf.predict(X_train)

print("\naccuracy_score(y_train, y_pred): \n", accuracy_score(y_train, y_pred))

print("\nOkay, 89.5% accuracy on MNIST is pretty bad. \n"
"This linear model is certainly too simple for MNIST, \n"
"but perhaps we just needed to scale the data first:\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

lin_clf = LinearSVC(random_state=42)
print("\nlin_clf.fit(X_train_scaled, y_train): \n", lin_clf.fit(X_train_scaled, y_train))

y_pred = lin_clf.predict(X_train_scaled)
print("\naccuracy_score(y_train, y_pred): \n", accuracy_score(y_train, y_pred))

print("\nThat's much better (we cut the error rate by about 25%), \n"
"but still not great at all for MNIST. If we want to use an SVM, \n"
"we will have to use a kernel. Let's try an SVC with an RBF kernel (the default).\n")

svm_clf = SVC(gamma="scale")
print("\nsvm_clf.fit(X_train_scaled[:10000], y_train[:10000]): \n", svm_clf.fit(X_train_scaled[:10000], y_train[:10000]))

y_pred = svm_clf.predict(X_train_scaled)
print("\naccuracy_score(y_train, y_pred): \n", accuracy_score(y_train, y_pred))

print("\nThat's promising, we get better performance even though we trained \n"
"the model on 6 times less data. Let's tune the hyperparameters by doing \n"
"a randomized search with cross validation. We will do this on a small dataset \n"
"just to speed up the process:\n")

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2, cv=3)
rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])

print("\nrnd_search_cv.best_estimator_: \n", rnd_search_cv.best_estimator_)

print("\nrnd_search_cv.best_score_: \n", rnd_search_cv.best_score_)

print("\nThis looks pretty low but remember we only trained the model on 1,000 instances. \n"
"Let's retrain the best estimator on the whole training set \n"
"(run this at night, it will take hours):\n")

print("\nrnd_search_cv.best_estimator_.fit(X_train_scaled, y_train): \n", rnd_search_cv.best_estimator_.fit(X_train_scaled, y_train))

y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
print("\naccuracy_score(y_train, y_pred): \n", accuracy_score(y_train, y_pred))

print("\nAh, this looks good! Let's select this model. Now we can test it on the test set:\n")

y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
print("\naccuracy_score(y_test, y_pred): \n", accuracy_score(y_test, y_pred))

print("\nNot too bad, but apparently the model is overfitting slightly. \n"
"It's tempting to tweak the hyperparameters a bit more \n"
"(e.g. decreasing C and/or gamma), but we would run the risk of overfitting \n"
"the test set. Other people have found that the hyperparameters C=5 and gamma=0.005 \n"
"yield even better performance (over 98% accuracy). By running the randomized search \n"
"for longer and on a larger part of the training set, you may be able to find this \n"
"as well.")