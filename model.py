from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


class Model:

    def __init__(self):
        self.vectorizer = None
        self.model = None

    def fit(self, comments_train: List[str], y_train):
        self.vectorizer = TfidfVectorizer()
        X_train = self.vectorizer.fit_transform(comments_train)
        self.model = SVC()
        self.model.fit(X_train, y_train)

    def predict(self, comments_test: List[str]):
        X_test = self.vectorizer.transform(comments_test)
        return self.model.predict(X_test)

    def score(self, comments_test: List[str], y_test):
        predictions = self.predict(comments_test)
        return accuracy_score(predictions, y_test)