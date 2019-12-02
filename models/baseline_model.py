from typing import List

import pandas as pd
import sklearn
from bs4 import BeautifulSoup as Soup
from sklearn.linear_model import LogisticRegression

from models.model import Model, Vectorizer
from pipeline.pipeline import evaluate_model
import numpy as np

raw_df = pd.read_csv("Reviews.csv", sep=',', quotechar='"')
df = raw_df.sample(n=10000)
texts = df["Text"]
texts = [Soup(text, features="html.parser").get_text() for text in texts]
helpfulnessNumerators = df["HelpfulnessNumerator"]
helpfulnessDenominators = df["HelpfulnessDenominator"]
scores = [helpfulnessDenominator < 5 or helpfulnessNumerator / helpfulnessDenominator < 0.7769 for
          helpfulnessNumerator, helpfulnessDenominator in
          zip(helpfulnessNumerators, helpfulnessDenominators)]


class LengthVectorizer:
    def fit_transform(self, reviews: List[str]):
        return self.transform(reviews)

    def transform(self, reviews: List[str]):
        return np.array([[len(review)] for review in reviews])


model = Model(vectorizer=Vectorizer(pca=False, base_model=LengthVectorizer()), model=LogisticRegression(solver="lbfgs"))

print(evaluate_model(model, texts, scores,
                     score=lambda actual, predicted: sklearn.metrics.fbeta_score(actual, predicted, 1)))
