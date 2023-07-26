import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import numpy as np
import os
import pandas as pd

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "deploy"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import googleapiclient.discovery

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

print("\n\n###############################################################")
print("And now let's use saved_model_cli to make predictions \n" 
"for the instances we just saved:")
print("let's start by creating the query.")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "annular-orb-349114-f326873bc791.json"
project_id = "annular-orb-349114"
model_id = "my_pima_model"
model_path = "projects/{}/models/{}".format(project_id, model_id)
model_path += "/versions/v0001/"
ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

print("\nmodel_path: \n", model_path)

def predict(X):
    input_data_json = {"signature_name": "serving_default", "instances": X.tolist()}
    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    print("\nresponse: \n", response)
    if "error" in response:
        raise RuntimeError(response["error"])
    return np.array([pred['dense_2'] for pred in response["predictions"]])

print("\nX_test: \n", X_test)
Y_probas = predict(X_test[:3])

print("\n\npredict(X_test[:3]): \n", np.round(Y_probas, 2))
print("2018250051 차수진")