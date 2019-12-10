from sklearn.feature_extraction.text import CountVectorizer

from models.WordEmbeddingModel import WordEmbeddingModel
from models.model import Model, Vectorizer
from pipeline.Pipeline import Pipeline


class RatingPredictorWordEmbedderPipeline(Pipeline):

    def __init__(self):
        super().__init__("Score", 0.5)

    def make_model(self):
        return Model(model=WordEmbeddingModel(), vectorizer=Vectorizer(pca=False, base_model=CountVectorizer()))

    def label_func(self, item):
        return 1 if item > 3 else 0


RatingPredictorWordEmbedderPipeline().evaluate()
