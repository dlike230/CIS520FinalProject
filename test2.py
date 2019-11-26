import numpy as np
from model import Model, Vectorizer
# import sklearn.scikit
# from sklearn.linear_model import, LinearRegression, LogisticRegression
from numpy import linalg as LA
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
import random
import os
import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import scipy.sparse

# path = "C://User//Owen//Desktop//sarcasm test//train-balanced-sarcasm.csv"
path = "Reviews.csv"
num_samples = None

df=pd.read_csv(path, sep=',', nrows=num_samples, quotechar = '"')

'''
Index(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
       'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text'],
      dtype='object')
'''

cols = ['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Summary', 'Text']



# axes = df.boxplot(by='UserId')
# plt.axes(axes)

ctr = 0
summ = 0

corr = df.corr()

print(corr)

# for index, row in df.iterrows():
# 	num = row['HelpfulnessNumerator']
# 	denom = row['HelpfulnessDenominator']
# 	if denom != 0:
# 		summ += num / denom
# 		ctr += 1

# print(summ / ctr)

plt.show()