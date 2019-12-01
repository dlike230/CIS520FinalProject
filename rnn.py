import os

import nltk
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class RNNEncoder:

    def __init__(self):
        self.transform = None
        self.vocab_size = -1
        self.padded_length = -1

    @staticmethod
    def samples_as_word_lists(samples):
        return [[word for sentence in nltk.sent_tokenize(train_string) for word in
                 nltk.word_tokenize(sentence)] for train_string in samples]

    def pad(self, encoding):
        return encoding + [0] * (self.padded_length - len(encoding))

    def _transform(self, samples_as_word_lists, word_mapping):
        return np.array(
            [self.pad([word_mapping[word] if word in word_mapping else len(word_mapping) + 1 for word in word_list]) for
             word_list in samples_as_word_lists])

    def fit_transform(self, train_strings):
        words_per_sample = RNNEncoder.samples_as_word_lists(train_strings)
        self.padded_length = max(len(sample) for sample in words_per_sample)
        word_set = {word for sample in words_per_sample for word in sample}
        word_mapping = {}
        for i, word in enumerate(word_set):
            word_mapping[word] = i + 1
        self.vocab_size = len(word_set)
        self.transform = lambda texts: self._transform(RNNEncoder.samples_as_word_lists(texts), word_mapping)

        return self._transform(words_per_sample, word_mapping)


class RNN:

    def __init__(self):
        self.encoder = RNNEncoder()
        self.model = None

    def fit(self, train_strings, y_train):
        encoded = self.encoder.fit_transform(train_strings)
        vocab_size = self.encoder.vocab_size
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size + 2, 16),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(1e-4),
                           metrics=['accuracy'])
        self.model.fit(x=encoded, y=np.array([1 if item else 0 for item in y_train]), epochs=10)

    def predict(self, reviews_test):
        return self.model.predict(self.encoder.transform(reviews_test))
