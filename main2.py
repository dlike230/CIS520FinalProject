import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as Soup
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import ElasticNet, LogisticRegression

raw_df = pd.read_csv("Reviews.csv", sep=',', nrows=10000, quotechar='"')
df = raw_df[raw_df["Score"] >= 2]
# df = raw_df
texts = df["Text"]
texts = [Soup(text, features="html.parser").get_text() for text in texts]
helpfulnessNumerators = df["HelpfulnessNumerator"]
helpfulnessDenominators = df["HelpfulnessDenominator"]
reviews = [score > 3 for score in df["Score"]]
scores = [helpfulnessNumerator - helpfulnessDenominator for helpfulnessNumerator, helpfulnessDenominator in
		  zip(helpfulnessNumerators, helpfulnessDenominators)]
helpfuls = [helpfulnessNumerator > 5 for helpfulnessNumerator in helpfulnessNumerators]
n_total = len(texts)
n_train = int(n_total * 0.5)

text_train = texts[:n_train]
text_test = texts[n_train:]

y_train = helpfuls[:n_train]
y_test = helpfuls[n_train:]

vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
dim_reducer = TruncatedSVD(n_components=1000)
X_train = dim_reducer.fit_transform(vectorizer.fit_transform(text_train))
X_test = dim_reducer.transform(vectorizer.transform(text_test))

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

