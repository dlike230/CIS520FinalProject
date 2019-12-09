import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression

from models.AutoEncoder import BagOfWordsAutoEncoder
from models.model import Model, Vectorizer
from pipeline.Pipeline import Pipeline


class RatingPredictorLogisticNoDimReduction(Pipeline):

    def __init__(self):
        super().__init__("Score", [sklearn.metrics.accuracy_score,
                                   lambda actual, predicted: sklearn.metrics.fbeta_score(actual, predicted, 1)],
                         0.5)

    def make_model(self):
        return Model(model=LogisticRegression(), vectorizer=Vectorizer(pca=True))

    def label_func(self, item):
        return 1 if item > 3 else 0


RatingPredictorLogisticNoDimReduction().evaluate()
