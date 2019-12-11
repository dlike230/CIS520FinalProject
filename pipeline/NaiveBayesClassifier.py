from models.model import Model, Vectorizer
from pipeline.Pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

class NaiveBayesClassifier(Pipeline):

    def __init__(self):
        super().__init__("Score", 0.5)

    def make_model(self):
        return Model(model=GaussianNB(), vectorizer=Vectorizer(pca=True, make_array=True))

    def label_func(self, item):
        return 1 if item > 3 else 0


NaiveBayesClassifier().evaluate()
