from typing import List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


class Vectorizer:

    def __init__(self, pca=True, base_model=TfidfVectorizer(lowercase=True), make_array=False):
        self.base_model = base_model
        self.dimensionality_reducer = None
        self.pca = pca
        self.make_array = make_array

    def fit_transform(self, reviews: List[str]):
        X_train = self.base_model.fit_transform(reviews)
        if self.pca:
            self.dimensionality_reducer = TruncatedSVD(n_components=1000)
            X_train = self.dimensionality_reducer.fit_transform(X_train)
        elif self.make_array:
            X_train = X_train.toarray()
        return X_train

    def transform(self, comments: List[str]):
        if self.pca:
            comments = self.dimensionality_reducer.transform(self.base_model.transform(comments))
        else:
            comments = self.base_model.transform(comments)
            if self.make_array:
                comments = comments.toarray()
        return comments


class Model:

    def __init__(self, vectorizer=Vectorizer(pca=True), model=SVC()):
        self.vectorizer = vectorizer
        self.model = model

    def fit(self, reviews_train: List[str], y_train):
        X_train = self.vectorizer.fit_transform(reviews_train) if self.vectorizer is not None else reviews_train
        self.model.fit(X_train, y_train)

    def predict(self, reviews_test: List[str]):
        X_test = self.vectorizer.transform(reviews_test) if self.vectorizer is not None else reviews_test
        return np.round(np.array(self.model.predict(X_test)))

    def get_params(self, deep = True):
        return {'model' : self.model, 'vectorizer' : self.vectorizer}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
