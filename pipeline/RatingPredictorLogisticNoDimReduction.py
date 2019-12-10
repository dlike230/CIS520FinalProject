import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression

from models.WordAutoEncoder import BagOfWordsAutoEncoder
from models.model import Model, Vectorizer
from pipeline.Pipeline import Pipeline


class RatingPredictorLogisticNoDimReduction(Pipeline):

    def __init__(self):
        super().__init__("Score", 0.5)

    def make_model(self):
        return Model(model=LogisticRegression(), vectorizer=Vectorizer(pca=False))

    def label_func(self, item):
        return 1 if item > 3 else 0


RatingPredictorLogisticNoDimReduction().evaluate()
