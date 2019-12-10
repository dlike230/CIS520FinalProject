from pandas import DataFrame
from sklearn.metrics import accuracy_score, fbeta_score, auc

from pipeline.fetch_data import fetch_data, get_df, extract_text


class Pipeline:
    def __init__(self, label_col, p_train, should_subsample=True, sample_size=10000):
        df = get_df()
        df[label_col] = df[label_col].apply(self.label_func)
        if should_subsample:
            part1: DataFrame = df[df[label_col] == 1].sample(n=sample_size // 2)
            part2: DataFrame = df[df[label_col] == 0].sample(n=sample_size // 2)
            df = part1.append(part2)
            df = df.sample(frac=1)
        else:
            df = df.sample(n=sample_size)
        self.text_data = extract_text(df)
        self.labels = df[label_col]
        self.metrics = [accuracy_score, lambda a, b: fbeta_score(a, b, 1)]
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
        n_train = int(self.p_train * len(self.text_data))
        reviews_train = self.text_data[:n_train]
        reviews_test = self.text_data[n_train:]
        labels_train = self.labels[:n_train]
        labels_test = self.labels[n_train:]
        model = self.make_model()
        model.fit(reviews_train, labels_train)
        predictions = model.predict(reviews_test)
        for i, score in enumerate(self.metrics):
            print("Metric", str(i) + ":", score(predictions, labels_test))
