import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf


class BagOfWordsAutoEncoder:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.encoder = None
        self.decoder = None
        self.model = None

    def fit_transform(self, reviews):
        vectors = self.vectorizer.fit_transform(reviews).toarray()

        print(vectors.shape)
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(vectors.shape[1],)))
        self.encoder.add(tf.keras.layers.Dense(100, activation="relu", use_bias=False))
        self.encoder.add(tf.keras.layers.Dense(50, activation="relu", use_bias=False))
        self.encoder.add(tf.keras.layers.Dense(2, use_bias=False))

        self.decoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(2,)))
        self.decoder.add(tf.keras.layers.Dense(50, input_shape=(2,), use_bias=False, activation="relu"))
        self.decoder.add(tf.keras.layers.Dense(100, use_bias=False, activation="relu"))
        self.decoder.add(tf.keras.layers.Dense(vectors.shape[1], use_bias=False))

        self.model = tf.keras.Sequential([self.encoder, self.decoder])
        self.model.compile(loss="mean_squared_error", optimizer="sgd")
        self.model.fit(x=vectors, y=vectors, epochs=3)
        return self._transform(vectors)

    def _transform(self, vectors):
        return self.encoder.predict(vectors)

    def reconstruction_error(self, reviews):
        vectors = self.vectorizer.transform(reviews).toarray()
        encoded = self._transform(vectors)
        decoded = self.decoder.predict(encoded)
        return np.linalg.norm(decoded, vectors) / len(vectors)

    def transform(self, reviews):
        return self._transform(self.vectorizer.transform(reviews).toarray())
