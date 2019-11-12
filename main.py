import numpy as np
# import sklearn.scikit
# from sklearn.linear_model import LinearRegression, LogisticRegression
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
import os

# path = "C://User//Owen//Desktop//sarcasm test//train-balanced-sarcasm.csv"
path = "train-balanced-sarcasm.csv"

data = np.loadtxt(path, dtype = 'str', delimiter = ',', skiprows = 1, usecols = (0, 1), max_rows = 1000)

labels = data[:, 0].astype('int32')
sentences = data[: , 1]

print(labels)
print(sentences)

