# vect = TfidfVectorizer(lowercase=True)
# X = vect.fit_transform(texts)
# vals = metrics.graph_eigenvalues(X)
# print(vals)
# _, frobenii = metrics.graph_reconstruction(X, delta = 25, max_components = 5000, print_progress = True)
# print(frobenii)

"""vectorizer = Vectorizer(pca=True, base_model=TfidfVectorizer(lowercase=True))
# model = Model(vectorizer=vectorizer, model=LogisticRegression(solver="lbfgs", max_iter = 10000))
model = Model(vectorizer=vectorizer, model=SVC(C=2))"""
import sklearn

from pipeline.Pipeline import Pipeline
from models.Rnn import RNN


class RatingPredictorWordBasedRNNPipeline(Pipeline):

    def __init__(self):
        super().__init__("Score", 0.5)

    def make_model(self):
        return RNN(encode_words=True)

    def label_func(self, item):
        return 1 if item > 3 else 0


RatingPredictorWordBasedRNNPipeline().evaluate()
