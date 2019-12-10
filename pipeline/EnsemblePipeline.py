from models.model import Model, Vectorizer
from pipeline.Pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from models.bert import BERT
from sklearn.linear_model import LogisticRegression

class EnsemblePipeline(Pipeline):

    def __init__(self, models, names):
        super().__init__("Score", 0.5)
        self.models = models
        self.names = names

    def make_model(self):
        return Model(model=VotingClassifier(list(zip(self.names, self.models)), voting='hard'), vectorizer=None)

    def label_func(self, item):
        return 1 if item > 3 else 0

EnsemblePipeline([BERT(), Model(model=LogisticRegression(), vectorizer=Vectorizer(pca=False)), Model(model=AdaBoostClassifier(), vectorizer=Vectorizer(pca=False))],
                  ['bert', 'log_reg', 'ada_boost']).evaluate()
#
# EnsemblePipeline([Model(model=LogisticRegression(), vectorizer=Vectorizer(pca=False))], ['log']).evaluate()
