import numpy as np
import matplotlib.pyplot as plt
print("2018250051 차수진")
from sklearn import datasets
iris = datasets.load_iris()
X = iris['data'][:,(2,3)]  # 꽃잎 길이, 꽃잎 넓이
y = iris['target']

# Add Bias
X_with_bias = np.c_[np.ones([len(X),1]),X]
# len(X) 개수 만큼 1로 채워진 [ ], [ ]......[ ] array
# 2열 -> 3열로 늘어남

# 결과를 일정하게 하기 위해, random seed 배정
np.random.seed(2042)

# 원래면 sklearn의 train_test_split을 사용하지만, 직접 만들면서 원리 이해하기
test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

rnd_indices = np.random.permutation(total_size)    # 150을 무작위로 섞음

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:-test_size]]  # test_size 만큼 남겨둠
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]

# 클래스를 OneHot Vector로 바꾸기
def to_one_hot(y):
    n_classes = y.max()+1    # 0,1,2 라 max=2 / +1 하면 classes 개수
    m = len(y)                     # 총 150개의 라벨들
    y_one_hot = np.zeros((m,n_classes))
    y_one_hot[np.arange(m),y] = 1   # index의 행중에 y값을 1로 치환
    return y_one_hot

print("\ny_train[:10]: ")
print( y_train[:10] )
print( "\nto_one_hot(y_train[:10]: ")
print( to_one_hot(y_train[:10]) )  # one hot encoding이 된 것을 확인할 수 있다.

# 라벨 전부를 onehot encoding하기
y_train_one_hot = to_one_hot(y_train)
y_valid_one_hot = to_one_hot(y_valid)
y_test_one_hot = to_one_hot(y_test)

# Softmax 함수 만들기

def softmax(logits):
    exps = np.exp(logits)
    exp_sums = np.sum(exps,axis=1,keepdims=True)   # axis=1   ->  가장 안쪽의 [ ] 안의 성분의 합 / 각 exps들의 합
    return exps/exp_sums

# 입력과 출력의 갯수 정하기
n_inputs = X_train.shape[1]   # (90,3) 인데 1로 인덱싱 ==3
n_outputs = len(np.unique(y_train))  # y_train값을 중복되지 않는 값들을 출력  3

eta = 0.01
n_iteration = 5001
m = len(X_train)
epsilon = 1e-7 # ε : 입실론     nan값을 피하기 위해 logPi에 추가.

Theta = np.random.randn(n_inputs,n_outputs)
print("\n소프트맥스 모델을 훈련: ")
for i in range(n_iteration):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)
    loss = -np.mean(np.sum(y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
    error = Y_proba - y_train_one_hot
    if i % 500 == 0:
        print(i,loss)
    gradients = 1/m * X_train.T.dot(error)
    Theta = Theta - eta * gradients

# 모델 파라미터 확인
print("\nTheta: ")
print( Theta )

# 검증 세트에 대한 정확도 확인
logits = X_valid.dot(Theta)
y_proba = softmax(logits)
y_predict = np.argmax(y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_valid)
print("\naccuracy_score: ")
print( accuracy_score )
# 모델이 매우 잘 작동하는 것으로 확인됨
# L2규제를 추가해보자

eta = 0.1
n_iteration = 5001
m = len(X_train)
epilson = 1e-7
alpha = 0.1  # 규제 파라미터

Theta = np.random.randn(n_inputs,n_outputs)
print("\n학습률 eta증가, 손실에 l2패널치 추가, 그래디언트에 항 추가 소프트맥스 모델을 훈련: ")
for i in range(n_iteration):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)
    entropy_loss = -np.mean(np.sum(y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = entropy_loss + alpha * l2_loss
    error = Y_proba - y_train_one_hot
    if i % 500 == 0:
        print(i,loss)
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1,n_outputs]), alpha * Theta[1:]]
    Theta = Theta - eta * gradients

# l2 패널티 때문에 손실이 더 커보인다. 모델이 더 잘 작동하는지 확인해보자
logits = X_valid.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba,axis=1)

accuracy_score = np.mean(y_predict == y_valid)
print("\naccuracy_score: ")
print( accuracy_score )
# 더 성능이 좋은 모델이 되었다.

# 조기 종료 추가
eta = 0.1
m = len(X_train)
iteration = 5001
epsilon = 1e-7
alpha = 0.1
best_loss = np.infty

Theta = np.random.randn(n_inputs, n_outputs)
print("\n 조기 종료 추가 소프트맥스 모델을 훈련: ") #매 반복에서 검증 세트에 대한 손실을 계산해서 오차가 증가하기 시작할 때 멈춤
for i in range(iteration):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)
    entropy_loss = -np.mean(np.sum(y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
    l2_loss = 1 / 2 * np.sum(np.square(Theta[1:]))
    loss = entropy_loss + alpha * l2_loss
    error = Y_proba - y_train_one_hot
    gradients = 1 / m * X_train.T.dot(error) + np.r_[(np.zeros([1, n_outputs]), alpha * Theta[1:])]
    Theta = Theta - eta * gradients

    logits = X_valid.dot(Theta)
    Y_proba = softmax(logits)
    xentropy_loss = -np.mean(np.sum(y_valid_one_hot * np.log(Y_proba + epsilon), axis=1))
    l2_loss = 1 / 2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    if iteration % 500 == 0:
        print(i, loss)
    if loss < best_loss:
        best_loss = loss
    else:
        print(i - 1, best_loss)
        print(i, loss, "Early Stopping!")
        break

logits = X_valid.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_valid)
print("\naccuracy_score: ")
print( accuracy_score )

# 학습이 더 빠르게 종료되었다.

x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]
X_new_with_bias = np.c_[np.ones([len(X_new), 1]), X_new]

logits = X_new_with_bias.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

zz1 = Y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()

# 최종 테스트 데이터 예측
logits = X_test.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_test)
print("\naccuracy_score: ")
print(accuracy_score)

# 매우 정확한 예측도를 보였다.
