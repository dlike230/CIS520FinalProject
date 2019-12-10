import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

from models.WordAutoEncoder import BagOfWordsAutoEncoder
from models.model import Model, Vectorizer
from pipeline.Pipeline import Pipeline


class RatingPredictorSVM(Pipeline):

    def __init__(self):
        super().__init__("Score", 0.5)

    def make_model(self):
        return [("SVM: pca: %s, C: %s" % (pca, val), Model(model=SVC(C=val),
                                                                vectorizer=Vectorizer(pca=pca == 1))) for pca in
                range(2) for val in [0.25, 0.5, 1, 1.5, 2, 4]]

    def label_func(self, item):
        return 1 if item > 3 else 0


RatingPredictorSVM().evaluate()
