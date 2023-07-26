import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import tensorflow as tf
from tensorflow import keras
assert tf.__version__

import numpy as np
import os

np.random.seed(42)
tf.random.set_seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "deploy"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

print("2018250051 차수진")
print("\n\n##########################################################")
print("Deploying TensorFlow models to TensorFlow Serving (TFS)")
print("Save/Load a SavedModel")

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full[..., np.newaxis].astype(np.float32) / 255.
X_test = X_test[..., np.newaxis].astype(np.float32) / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_new = X_test[:3]

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28, 1]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()

model.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1e-2),
                metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

print("\nnp.round(model.predict(X_new), 2): \n", np.round(model.predict(X_new), 2))

print("\nExporting SavedModels: ")

model_version = "0001"
model_name = "my_mnist_model"
model_path = os.path.join(model_name, model_version)
print("\nmodel_path: \n", model_path)

tf.saved_model.save(model, model_path)

for root, dirs, files in os.walk(model_name):
    indent = ' ' * root.count(os.sep)
    print('{}{}/'.format(indent, os.path.basename(root)))
    for filename in files:
        print('{}{}'.format(indent + ' ', filename))

print("\nLet's write the new instances to a npy file so we can pass them easily to our model:")

np.save("my_mnist_tests.npy", X_new)
input_name = model.input_names[0]
print("\ninput_name: \n", input_name)
print("\n\nAnd now let's use saved_model_cli to make predictions for the instances we just saved:")

np.round([[1.1739199e-04, 1.1239604e-07, 6.0210604e-04, 2.0804715e-03, 2.5779348e-06,
6.4079795e-05, 2.7411186e-08, 9.9669880e-01, 3.9654213e-05, 3.9471846e-04],
[1.2294615e-03, 2.9207937e-05, 9.8599273e-01, 9.6755642e-03, 8.8930705e-08,
2.9156188e-04, 1.5831805e-03, 1.1311053e-09, 1.1980456e-03, 1.1113169e-07],
[6.4066830e-05, 9.6359509e-01, 9.0598064e-03, 2.9872139e-03, 5.9552520e-04,
3.7478798e-03, 2.5074568e-03, 1.1462728e-02, 5.5553433e-03, 4.2495009e-04]], 2)

print("\n\n##########################################################")
print("TensorFlow Serving")

os.environ["MODEL_DIR"] = os.path.split(os.path.abspath(model_path))[0]

import json
input_data_json = json.dumps({
            "signature_name": "serving_default",
            "instances": X_new.tolist(),
})

print('\nrepr(input_data_json)[:1500] + "...": \n', repr(input_data_json)[:1500] + "...")

print("\nNow let's use TensorFlow Serving's REST API to make predictions:")

import requests

SERVER_URL = 'http://localhost:8501/v1/models/my_mnist_model:predict'
response = requests.post(SERVER_URL, data=input_data_json)
response.raise_for_status() # raise an exception in case of error
response = response.json()

print("\nresponse.keys(): \n",response.keys())

y_proba = np.array(response["predictions"])
print("\ny_proba.round(2): \n", y_proba.round(2))

print("\n\n##########################################################")
print("Deploying a new model version")

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28, 1]),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1e-2),
                metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

model_version = "0002"
model_name = "my_mnist_model"
model_path = os.path.join(model_name, model_version)
print("\nmodel_path: \n", model_path)

tf.saved_model.save(model, model_path)

for root, dirs, files in os.walk(model_name):
    indent = ' ' * root.count(os.sep)
    print('{}{}/'.format(indent, os.path.basename(root)))
    for filename in files:
        print('{}{}'.format(indent + ' ', filename))