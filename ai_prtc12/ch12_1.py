
import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import tensorflow as tf
assert tf.__version__

import numpy as np
import os

np.random.seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

print("2018250051 차수진")
print("\n\nBuilding an Image Classifier: \n")

import tensorflow as tf
from tensorflow import keras

print("tf.__version__: ",tf.__version__)

print("keras.__version__: ",keras.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print("\nX_train_full.shape: \n",X_train_full.shape)
print("X_train_full.dtype: \n",X_train_full.dtype)

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.
plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()

print("\ny_train: \n", y_train)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print("\nclass_names[y_train[0]]: \n", class_names[y_train[0]])

print("\nX_valid.shape: \n", X_valid.shape)

print("\nX_test.shape: \n", X_test.shape)

n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
save_fig('fashion_mnist_plot', tight_layout=False)
plt.show()

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.Dense(300, activation="relu"),
keras.layers.Dense(100, activation="relu"),
keras.layers.Dense(10, activation="softmax")
])

print("\n\nmodel.layers: \n", model.layers)

print("\n\nmodel.summary(): \n", model.summary())

hidden1 = model.layers[1]
print("\nhidden1.name: \n", hidden1.name)

print("\nmodel.get_layer(hidden1.name) is hidden1: \n", model.get_layer(hidden1.name) is hidden1)

weights, biases = hidden1.get_weights()

print("\nweights: \n", weights)

print("\nweights.shape: \n", weights.shape)

print("\nweights.shape: \n", weights.shape)

print("\nbiases.shape: \n", biases.shape)

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

print("\nhistory.params: \n", history.params)

print("\nhistory.epoch: \n", history.epoch)

print("\nhistory.history.keys(): \n", history.history.keys())

import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
save_fig("keras_learning_curves_plot")
plt.show()

print("\n\nmodel.evaluate(X_test, y_test): \n", model.evaluate(X_test, y_test))

X_new = X_test[:3]
y_proba = model.predict(X_new)
print("\ny_proba.round(2): \n", y_proba.round(2))
#model.predict_classes(X_new)는 삭제될 예정입니다. 대신 np.argmax(model.predict(X_new), axis=-1)를 사용하세요.
y_pred = np.argmax(model.predict(X_new), axis=-1)
print("\ny_pred: \n", y_pred)

print("\nnp.array(class_names)[y_pred]: \n", np.array(class_names)[y_pred])

y_new = y_test[:3]
print("\ny_new: \n", y_new)

plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(class_names[y_test[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
save_fig('fashion_mnist_images_plot', tight_layout=False)
plt.show()

print("\n\n###############################################")
print("\nRegression MLP: ")
print("\nLet's load, split and scale the California housing dataset " "\n(the original one, not the modified one as in chapter 2):")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)

plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

print("\ny_pred: \n", y_pred)