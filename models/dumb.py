class Dumb:
    def __init__(self):
        self.model = None

    def fit(self, train_strings, y_train):
        c1 = sum(i for i in y_train if i == 0)
        c2 = len(y_train) - c1
        self.model = max(c1, c2)

    def predict(self, reviews_test):
        return [self.model] * len(reviews_test)
