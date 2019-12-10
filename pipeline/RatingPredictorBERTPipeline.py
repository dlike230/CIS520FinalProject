from pipeline.Pipeline import Pipeline
from models.bert import BERT


class RatingPredictorBERTPipeline(Pipeline):

    def __init__(self):
        super().__init__("Score", 0.5)

    def make_model(self):
        return BERT()

    def label_func(self, item):
        return 1 if item > 3 else 0


RatingPredictorBERTPipeline().evaluate()
