from models.dumb import Dumb
from pipeline.Pipeline import Pipeline


class RatingPredictorDumb(Pipeline):

    def __init__(self):
        super().__init__("Score", 0.5)

    def make_model(self):
        return Dumb()

    def label_func(self, item):
        return 1 if item > 3 else 0


RatingPredictorDumb().evaluate()
