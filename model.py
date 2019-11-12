from typing import List

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np


class Vectorizer:

    def __init__(self):
        self.base_model = None
        self.dimensionality_reducer = None

    def fit_transform(self, comments: List[str]):
        self.base_model = TfidfVectorizer(max_features=10000)
        X_train = self.base_model.fit_transform(comments)
        self.dimensionality_reducer = PCA()
        X_train = self.dimensionality_reducer.fit_transform(X_train)
        return X_train

    def transform(self, comments: List[str]):
        self.dimensionality_reducer.transform(self.base_model.transform(comments))

class Model:

    def __init__(self):
        self.vectorizer = None
        self.model = None

    def fit(self, comments_train: List[str], y_train):
        self.vectorizer = Vectorizer()
        X_train = self.vectorizer.fit_transform(comments_train)
        self.model = SVC()
        self.model.fit(X_train, y_train)

    def predict(self, comments_test: List[str]):
        X_test = self.vectorizer.transform(comments_test)
        return self.model.predict(X_test)

    def score(self, comments_test: List[str], y_test):
        predictions = self.predict(comments_test)
        return accuracy_score(predictions, y_test)