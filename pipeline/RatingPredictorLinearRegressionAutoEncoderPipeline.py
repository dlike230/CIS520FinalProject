import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression

from models.WordAutoEncoder import BagOfWordsAutoEncoder
from models.model import Model
from pipeline.Pipeline import Pipeline


class RatingPredictorLinearRegressionAutoEncoderPipeline(Pipeline):

    def __init__(self):
        super().__init__("Score", 0.5)

    def make_model(self):
        return Model(model=LogisticRegression(), vectorizer=BagOfWordsAutoEncoder(num_epochs=1))

    def label_func(self, item):
        return 1 if item > 3 else 0


RatingPredictorLinearRegressionAutoEncoderPipeline().evaluate()
