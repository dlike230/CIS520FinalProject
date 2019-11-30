import os

import nltk
import tensorflow as tf
import numpy as np
from tensorflow_core.python.framework.tensor_shape import TensorShape
from tensorflow_datasets.core.features import FeaturesDict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class RNNEncoder:

    def __init__(self):
        self.encoder = None
        self.vocab_size = -1

    def fit_transform(self, train_strings, train_labels):
        word_set = {word for string in train_strings for tokenized_sent in nltk.sent_tokenize(string) for word in
                    nltk.word_tokenize(tokenized_sent)}
        word_mapping = {}
        for i, word in enumerate(word_set):
            word_mapping[word] = i
        self.vocab_size = len(word_set)
        self.encoder = lambda text: [word_mapping[word] for sentence in nltk.sent_tokenize(text) for word in
                                     nltk.word_tokenize(sentence)]

        def generator():
            return zip(
                [tf.data.Dataset.from_tensor_slices(self.encoder(train_string)) for train_string in train_strings],
                train_labels)

        return tf.data.Dataset.from_generator(generator, output_types=(tf.int32, tf.int32),
                                              output_shapes=(TensorShape([None, None]), TensorShape([None])))

    def transform(self, strings):
        return tf.constant([self.encoder(string) for string in strings])


class RNN:

    def __init__(self):
        self.encoder = RNNEncoder()
        self.model = None

    def fit(self, train_strings, y_train):
        encoded = self.encoder.fit_transform(train_strings, y_train)
        vocab_size = self.encoder.vocab_size
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(1e-4),
                           metrics=['accuracy'])
        self.model.fit(encoded, epochs=10)

    def predict(self, reviews_test):
        return self.model.predict(self.encoder.transform(reviews_test))
