print("2018250051 차수진")
import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 그림 저장
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

#내 닷홈에서 데이터 가져오는 부분
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "http://soojin9911.dothome.co.kr/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#데이터의 구조 출력하기
housing = load_housing_data()
print("### housing.head():\n")
print(housing.head())

print("\n### housing.info():\n")
print(housing.info())

print("\n### housing[ocean_proximity].value_counts():\n")
print(housing["ocean_proximity"].value_counts())

print("\n### housing.describe():\n")
print(housing.describe())

print("\n\n### Histogram plots:\n")
housing.hist(bins=50, figsize=(20, 15))
save_fig("01_attribute_histogram_plots")
plt.show()

print("\n\n##############################\n")
print("Create a Test Set\n")

print("\n\nFor illustration only. Sklearn has train_test_split()\n")
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

print("\n\n###############################################\n")

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print("\n\ntest_set.head():\n", test_set.head())

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5 )
housing["income_cat"].where(housing["income_cat"] <5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print("\n\n###############################################\n")
print("Discover and visualize the data to gain insights:\n")
housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("02_bad_visualization_plot")
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("03_better_visualization_plot")
plt.show()

print("\n\nHousing prices:\n")

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
save_fig("04_housing_prices_scatterplot")
plt.show()

corr_matrix = housing.corr()
print("\nCORR_MATRIX:\n")
print(corr_matrix["median_house_value"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("06_scatter_matrix_plot")
plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.axis([0, 16, 0, 550000])
save_fig("07_income_vs_house_value_scatterplot")
plt.show()

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
print("\nCORR_MATRIX AGAIN:\n")
print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()

print("\n\nhousing.describe():\n")
print(housing.describe())

