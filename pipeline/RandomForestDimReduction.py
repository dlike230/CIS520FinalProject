from sklearn.ensemble import RandomForestClassifier

from models.model import Model, Vectorizer
from pipeline.Pipeline import Pipeline


class RandomForestDimReduction(Pipeline):

    def __init__(self):
        super().__init__("Score", 0.5)

    def make_model(self):
        return Model(model=RandomForestClassifier(), vectorizer=Vectorizer(pca=True))

    def label_func(self, item):
        return 1 if item > 3 else 0


RandomForestDimReduction().evaluate()
