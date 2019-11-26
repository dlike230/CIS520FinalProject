from typing import List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


class Vectorizer:

	def __init__(self, pca=True, base_model=TfidfVectorizer(lowercase=True)):
		self.base_model = base_model
		self.dimensionality_reducer = None
		self.pca = pca

	def fit_transform(self, reviews: List[str]):
		X_train = self.base_model.fit_transform(reviews)
		if self.pca:
			self.dimensionality_reducer = TruncatedSVD(n_components=1000)
			X_train = self.dimensionality_reducer.fit_transform(X_train)
		return X_train

	def transform(self, comments: List[str]):
		if self.pca:
			return self.dimensionality_reducer.transform(self.base_model.transform(comments))
		else:
			return self.base_model.transform(comments)


class Model:

	def __init__(self, vectorizer=Vectorizer(pca=True), model=SVC()):
		self.vectorizer = vectorizer
		self.model = model

	def fit(self, reviews_train: List[str], y_train):
		X_train = self.vectorizer.fit_transform(reviews_train)
		self.model.fit(X_train, y_train)

	def predict(self, reviews_test: List[str]):
		X_test = self.vectorizer.transform(reviews_test)
		return self.model.predict(X_test)

	def score(self, comments_test: List[str], y_test, score=accuracy_score):
		predictions = self.predict(comments_test)
		return score(predictions, y_test)


def evaluate_model(model: Model, reviews, labels, p_train=0.5, score=accuracy_score):
	n_train = int(p_train * len(reviews))
	reviews_train = reviews[:n_train]
	reviews_test = reviews[n_train:]
	labels_train = labels[:n_train]
	labels_test = labels[n_train:]
	model.fit(reviews_train, labels_train)
	predictions = model.predict(reviews_test)
	return score(predictions, labels_test)
