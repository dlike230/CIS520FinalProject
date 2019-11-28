import tensorflow as tf
import nltk
from tensorflow_core.python.framework.dtypes import int64
from tensorflow_core.python.framework.tensor_shape import TensorShape
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.layers.embeddings import Embedding
from tensorflow_core.python.keras.layers.recurrent import LSTM
from tensorflow_core.python.keras.layers.wrappers import Bidirectional
from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

import os
from tensorflow_datasets.core.features.text import TokenTextEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class RNNEncoder:

    def __init__(self):
        self.encoder = None
        self.vocab_size = -1

    def fit_transform(self, train_strings, train_labels):
        word_set = {word for string in train_strings for tokenized_sent in nltk.sent_tokenize(string) for word in
                    nltk.word_tokenize(tokenized_sent)}
        self.vocab_size = len(word_set)
        self.encoder = TokenTextEncoder(word_set)

        print(self.encoder.encode("chicken is good"))
        def generator():
            yield from zip(train_strings, train_labels)

        train_data = tf.data.Dataset.from_generator(generator, (tf.string, tf.int64),
                                                    (TensorShape([None]), TensorShape([None])))
        return train_data.map(lambda x, y: (self.encoder.encode(str(x)), y))

    def transform(self, strings):
        def generator():
            yield from (self.encoder.encode(string) for string in strings)

        return tf.data.Dataset.from_generator(generator, (int64, int64),
                                              (TensorShape([]), TensorShape([None])))


class RNN:

    def __init__(self):
        self.encoder = RNNEncoder()
        self.model = None

    def fit(self, train_strings, y_train):
        encoded = self.encoder.fit_transform(train_strings, y_train)
        vocab_size = self.encoder.vocab_size
        self.model = Sequential([
            Embedding(vocab_size, 64),
            Bidirectional(LSTM(64)),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',
                           optimizer=Adam(1e-4),
                           metrics=['accuracy'])
        self.model.fit(encoded, epochs=10)

    def predict(self, reviews_test):
        return self.model.predict(self.encoder.transform(reviews_test))
