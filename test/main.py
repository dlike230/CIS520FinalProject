import numpy as np
from models.model import Vectorizer
# import sklearn.scikit
# from sklearn.linear_model import, LinearRegression, LogisticRegression
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn import preprocessing
import autosklearn.classification
from sklearn.metrics import accuracy_score

# path = "C://User//Owen//Desktop//sarcasm test//train-balanced-sarcasm.csv"
path = "train-balanced-sarcasm.csv"
num_samples = 5000

use_cols = (0, 1, 2, 3, 4, 9)

df=pd.read_csv(path, sep=',',header=None, skiprows=1, usecols=use_cols, nrows=num_samples, quotechar = '"')

data = df.values


# cols = [1, 2]


y = data[1:, 0].astype('int32')
X = data[1:, 1:]


class_columns = [1, 2] #username, subreddit
text_columns = [0, 4]




# le = preprocessing.LabelEncoder()

# for i in class_columns:
# 	X[: , i] = le.fit_transform(X[:, i])


ohe = preprocessing.OneHotEncoder()
cats = ohe.fit_transform(X[:, class_columns])

svd = TruncatedSVD(n_components=1000)
cats = svd.fit_transform(cats)

text = X[:, text_columns]



print(text)

vect = Vectorizer(pca = True)

newText = []

for col in text.T:
	newText.append(vect.fit_transform(col))

text = np.hstack(newText)

# np.random.seed(100)
# p = np.random.permutation(len(X))
# X, y = X[p], y[p]

# X = scipy.sparse.hstack((text, cats))

other = X[:, [i for i in range(X.shape[1]) if i not in text_columns and i not in class_columns]]

X = np.concatenate((text, cats), axis = 1)

print(X)

X_train, y_train = X[:num_samples // 2], y[:num_samples // 2]
X_test, y_test = X[num_samples // 2:], y[num_samples // 2:]

# model = Model()

# model.fit(X_train, y_train, text_indices = [0, 4], categorical_indices = class_columns)

# print(model.score(X_test, y_test))

clsf = autosklearn.classification.AutoSklearnClassifier()
clsf.fit(X_train, y_train)
y_test = clsf.predict(X_test)
accuracy_score(predictions, y_test)


# print(labels)
# print(sentences)
