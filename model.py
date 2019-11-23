from typing import List

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import scipy.sparse
import numpy as np


class Vectorizer:

    def __init__(self, pca = False):
        self.base_model = None
        self.dimensionality_reducer = None
        self.pca = pca

    def fit_transform(self, comments: List[str]):
        self.base_model = TfidfVectorizer(lowercase = False)
        X_train = self.base_model.fit_transform(comments)
        if self.pca:
            self.dimensionality_reducer = TruncatedSVD(n_components=1000)
            X_train = self.dimensionality_reducer.fit_transform(X_train)
        return X_train

    def transform(self, comments: List[str]):
        if self.pca:
            return self.dimensionality_reducer.transform(self.base_model.transform(comments))
        else:
            return self.base_model.transform(comments)

class Model:

    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.text_indices = None

    def fit(self, comments_train: List[str], y_train, text_indices : List[int] = None, categorical_indices = None):
        self.vectorizer = Vectorizer(pca = True)
                # X_train = np.append(X_train, x);
        # if text_indices is not None:
        #     self.text_indices = text_indices
        # X_train = self.vectorize_columns(comments_train)
        # print(X_train.shape)
        X_train = comments_train
        self.model = SVC()
        self.model.fit(X_train, y_train)


    def vectorize_columns(self, comments_train):
        n, p = comments_train.shape
        if self.text_indices is None:
            X_train = self.vectorizer.fit_transform(comments_train)
        else:
            X_train = comments_train[:, [i for i in range(p) if i not in self.text_indices]].astype('float64')
            for col in comments_train[ :, self.text_indices].T:
                x = self.vectorizer.fit_transform(col.astype('U'))
                if X_train.shape[1] == 0:
                    X_train = x
                else:
                    X_train = np.hstack((X_train, x))
        return X_train

    def predict(self, comments_test: List[str]):
        # X_test = self.vectorize_columns(comments_test)
        X_test = comments_test
        return self.model.predict(X_test)

    def score(self, comments_test: List[str], y_test):
        predictions = self.predict(comments_test)
        return accuracy_score(predictions, y_test)