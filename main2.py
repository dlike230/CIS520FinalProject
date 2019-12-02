import pandas as pd
import sklearn
from bs4 import BeautifulSoup as Soup
from sklearn.feature_extraction.text import TfidfVectorizer

import metrics
from model import Model, evaluate_model

from rnn import RNN

raw_df = pd.read_csv("Reviews.csv", sep=',', quotechar='"')
df = raw_df.sample(n=10000)  # , random_state = 100) #seed for consistency
# df = raw_df
texts = df["Text"]
texts = [Soup(text, features="html.parser").get_text() for text in texts]
helpfulnessNumerators = df["HelpfulnessNumerator"]
helpfulnessDenominators = df["HelpfulnessDenominator"]
ratings = df["Score"]
scores = [1 if rating > 3 else 0 for rating in ratings]

# vect = TfidfVectorizer(lowercase=True)
# X = vect.fit_transform(texts)
# vals = metrics.graph_eigenvalues(X)
# print(vals)
# _, frobenii = metrics.graph_reconstruction(X, delta = 25, max_components = 5000, print_progress = True)
# print(frobenii)

"""vectorizer = Vectorizer(pca=True, base_model=TfidfVectorizer(lowercase=True))
# model = Model(vectorizer=vectorizer, model=LogisticRegression(solver="lbfgs", max_iter = 10000))
model = Model(vectorizer=vectorizer, model=SVC(C=2))"""
model = Model(vectorizer=None, model=RNN())

print(evaluate_model(model, texts, scores,
                     score=lambda actual, predicted: sklearn.metrics.fbeta_score(actual, predicted, 1)))
