import pandas as pd
from bs4 import BeautifulSoup as Soup
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet

df = pd.read_csv("Reviews.csv", sep=',', nrows=5000, quotechar='"')
texts = df["Text"]
texts = [Soup(text).get_text() for text in texts]
helpfulnessNumerators = df["HelpfulnessNumerator"]
helpfulnessDenominators = df["HelpfulnessNumerator"]
scores = [helpfulnessNumerator - helpfulnessDenominator for helpfulnessNumerator, helpfulnessDenominator in
		  zip(helpfulnessNumerators, helpfulnessDenominators)]
n_total = len(texts)
n_train = n_total * 0.5
n_test = n_total * 0.5

text_train = texts[:n_train]
text_test = texts[n_train:]

y_train = scores[:n_train]
y_test = scores[n_train:]

vectorizer = TfidfVectorizer(lowercase=False)
dim_reducer = PCA(n_components=1000)
X_train = dim_reducer.fit_transform(vectorizer.fit_transform(text_train))
X_test = dim_reducer.transform(vectorizer.transform(text_test))

model = ElasticNet()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

