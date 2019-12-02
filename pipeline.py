import pandas as pd
import sklearn
from bs4 import BeautifulSoup as Soup
from sklearn.metrics import accuracy_score

from model import Model
from rnn import RNN


class Pipeline:
    def __init__(self, label_col, metrics, p_train):
        raw_df = pd.read_csv("Reviews.csv", sep=',', quotechar='"')
        df = raw_df.sample(n=10000)
        texts = df["Text"]
        self.texts = [Soup(text, features="html.parser").get_text() for text in texts]
        column = df[label_col]
        self.labels = [self.label_func(item) for item in column]
        self.metrics = metrics
        self.p_train = p_train

    def label_func(self, item):
        """
        Takes an item from the label column in the dataset and generates a numerical label to input into the model
        :param item:
        :return:
        """
        raise Exception("Not implemented")

    def make_model(self):
        raise Exception("Not implemented")

    def evaluate(self):
        n_train = int(self.p_train * len(self.texts))
        reviews_train = self.texts[:n_train]
        reviews_test = self.texts[n_train:]
        labels_train = self.labels[:n_train]
        labels_test = self.labels[n_train:]
        model = self.make_model()
        model.fit(reviews_train, labels_train)
        predictions = model.predict(reviews_test)
        for i, score in enumerate(self.metrics):
            print("Metric", str(i) + ":", score(predictions, labels_test))
