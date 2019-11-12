import numpy as np
from model import Model
# import sklearn.scikit
# from sklearn.linear_model import LinearRegression, LogisticRegression
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
import os

# path = "C://User//Owen//Desktop//sarcasm test//train-balanced-sarcasm.csv"
path = "train-balanced-sarcasm.csv"
num_samples = 10000

data = np.loadtxt(path, dtype='str', delimiter=',', skiprows=1, usecols=(0, 1), max_rows=num_samples)

y = data[:, 0].astype('int32')
X = data[:, 1]

# np.random.seed(100)
# p = np.random.permutation(len(X))
# X, y = X[p], y[p]

X_train, y_train = X[:num_samples // 2], y[:num_samples // 2]
X_test, y_test = X[num_samples // 2:], y[num_samples // 2:]

model = Model()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

# print(labels)
# print(sentences)
