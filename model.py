from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


class Model:

    def __init__(self):
        self.vectorizer = None
        self.model = None

    def fit(self, comments_train, y_train):
        self.vectorizer = TfidfVectorizer()
        X_train = self.vectorizer.fit_transform(comments_train)
        self.model = SVC()
        self.model.fit(X_train, y_train)

    def predict(self, comments_test):
        X_test = self.vectorizer.transform(comments_test)
        return self.model.predict(X_test)