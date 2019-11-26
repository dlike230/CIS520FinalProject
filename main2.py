import pandas as pd
import numpy as np
import sklearn
from bs4 import BeautifulSoup as Soup
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.svm import SVC

from model import Model, Vectorizer, evaluate_model

raw_df = pd.read_csv("Reviews.csv", sep=',', quotechar='"')
df = raw_df.sample(n=10000) #, random_state = 100) #seed for consistency
# df = raw_df
texts = df["Text"]
texts = [Soup(text, features="html.parser").get_text() for text in texts]
helpfulnessNumerators = df["HelpfulnessNumerator"]
helpfulnessDenominators = df["HelpfulnessDenominator"]
scores = [helpfulnessDenominator < 1 or helpfulnessNumerator / helpfulnessDenominator < 0.7769 for
		  helpfulnessNumerator, helpfulnessDenominator in
		  zip(helpfulnessNumerators, helpfulnessDenominators)]

vectorizer = Vectorizer(pca=True, base_model=TfidfVectorizer(lowercase=True))
model = Model(vectorizer=vectorizer, model=LogisticRegression(solver="lbfgs"))
print(evaluate_model(model, texts, scores,
			   score=lambda actual, predicted: sklearn.metrics.fbeta_score(actual, predicted, 1)))
