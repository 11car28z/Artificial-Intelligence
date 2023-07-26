import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >="2.0"

import numpy as np
import os
import pandas as pd

np.random.seed(42)
tf.random.set_seed(42)

import matplotlib as mpl
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


from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('./datasets/diabetes.csv', sep=',')

print("\ndata.head(): \n", data.head())

data.describe()

data.info()
print("\n\nStep 2 - Prepare the data for the model building")

X = data.values[:,0:8]
y = data.values[:,8]

scaler = MinMaxScaler()
scaler.fit(X)

X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("\n\nStep 3 - Create and train the model")

inputs = keras.Input(shape=(8,))
hidden1 = Dense(12,activation='relu')(inputs)
hidden2 = Dense(8,activation='relu',)(hidden1)
output = Dense(1,activation='sigmoid')(hidden2)
model = keras.Model(inputs,output)

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=100,batch_size=16,verbose=0)

model.save("my_remote_pima_model.h5")

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss', color=color)
ax1.plot(history.history['loss'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.plot(history.history['accuracy'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

y_pred = model.predict(X_test[:3])
print("\ny_pred: \n", y_pred)

model_version = "0001"
model_name = "my_pima_model"
model_path = os.path.join(model_name, model_version)
print("\nmodel_path: \n", model_path)

tf.saved_model.save(model, model_path)
print("2018250051 차수진")