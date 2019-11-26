import pandas as pd
import numpy as np
import sklearn
from bs4 import BeautifulSoup as Soup
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import metrics

raw_df = pd.read_csv("Reviews.csv", sep=',', quotechar='"')
df = raw_df.sample(n=10000) #, random_state = 100) #seed for consistency
# df = raw_df
texts = df["Text"]
texts = [Soup(text, features="html.parser").get_text() for text in texts]
helpfulnessNumerators = df["HelpfulnessNumerator"]
helpfulnessDenominators = df["HelpfulnessDenominator"]
reviews = [score > 3 for score in df["Score"]]
scores = [helpfulnessDenominator < 1 or helpfulnessNumerator / helpfulnessDenominator < 0.7769 for
		  helpfulnessNumerator, helpfulnessDenominator in
		  zip(helpfulnessNumerators, helpfulnessDenominators)]
n_total = len(texts)
n_train = int(n_total * 0.5)

text_train = texts[:n_train]
text_test = texts[n_train:]

y_train = scores[:n_train]
y_test = scores[n_train:]

vectorizer = TfidfVectorizer(lowercase=True)
dim_reducer = TruncatedSVD(n_components=1000)
X_train = dim_reducer.fit_transform(vectorizer.fit_transform(text_train))
X_test = dim_reducer.transform(vectorizer.transform(text_test))

# X = vectorizer.fit_transform(texts)
# metrics.graph_reconstruction(X)

model = LogisticRegression(solver="lbfgs", max_iter = 10000)
model.fit(X_train, y_train)
print(sklearn.metrics.fbeta_score(y_test, model.predict(X_test), 1))
