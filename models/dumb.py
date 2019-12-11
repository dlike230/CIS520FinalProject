class Dumb:
    def __init__(self):
        self.model = None

    def fit(self, train_strings, y_train):
        c1 = 0
        for y in y_train:
            if y == 0:
                c1 += 1
        c2 = len(y_train) - c1
        self.model = 0 if c1 > c2 else 1

    def predict(self, reviews_test):
        return [self.model] * len(reviews_test)

    def get_params(self, deep = True):
        return {'model' : self.model}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
