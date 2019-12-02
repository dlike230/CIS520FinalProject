import os

import nltk
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class WordRNNEncoder:

    def __init__(self):
        self.transform = None
        self.vocab_size = -1
        self.padded_length = -1

    @staticmethod
    def samples_as_word_lists(samples):
        return [[word for sentence in nltk.sent_tokenize(train_string) for word in
                 nltk.word_tokenize(sentence)] for train_string in samples]

    def pad(self, encoding):
        if len(encoding) >= self.padded_length:
            return encoding[:self.padded_length]
        return encoding + [0] * (self.padded_length - len(encoding))

    def _transform(self, samples_as_word_lists, word_mapping):
        return np.array(
            [self.pad([word_mapping[word] if word in word_mapping else len(word_mapping) + 1 for word in word_list]) for
             word_list in samples_as_word_lists])

    def fit_transform(self, train_strings):
        words_per_sample = WordRNNEncoder.samples_as_word_lists(train_strings)
        self.padded_length = max(len(sample) for sample in words_per_sample)
        word_set = {word for sample in words_per_sample for word in sample}
        word_mapping = {}
        for i, word in enumerate(word_set):
            word_mapping[word] = i + 1
        self.vocab_size = len(word_set)
        self.transform = lambda texts: self._transform(WordRNNEncoder.samples_as_word_lists(texts), word_mapping)

        return self._transform(words_per_sample, word_mapping)


class CharacterRNNEncoder:

    def __init__(self):
        self.transform = None
        self.vocab_size = -1
        self.padded_length = -1

    def pad(self, encoding):
        if len(encoding) >= self.padded_length:
            return encoding[:self.padded_length]
        return encoding + [0] * (self.padded_length - len(encoding))

    def _transform(self, samples, character_mapping):
        return np.array(
            [self.pad(
                [character_mapping[character] if character in character_mapping else len(character_mapping) + 1 for
                 character in
                 characters]) for
             characters in samples])

    def fit_transform(self, train_strings):
        self.padded_length = max(len(sample) for sample in train_strings)
        character_set = {c for sample in train_strings for c in sample}
        character_mapping = {}
        for i, character in enumerate(character_set):
            character_mapping[character] = i + 1
        self.vocab_size = len(character_set)
        self.transform = lambda texts: self._transform(WordRNNEncoder.samples_as_word_lists(texts), character_mapping)

        return self._transform(train_strings, character_mapping)


class RNN:

    def __init__(self, encode_words=True):
        if encode_words:
            self.encoder = WordRNNEncoder()
        else:
            self.encoder = CharacterRNNEncoder()
        self.model = None

    def fit(self, train_strings, y_train):
        val_size = round(len(train_strings) * 0.5)
        train_strings, val_strings = train_strings[:val_size], train_strings[val_size:]
        X_train = self.encoder.fit_transform(train_strings)
        y_train, y_val = y_train[:val_size], y_train[val_size:]
        X_val = self.encoder.transform(val_strings)
        vocab_size = self.encoder.vocab_size
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size + 2, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(1e-4),
                           metrics=[
                               tf.keras.metrics.TruePositives(name='tp'),
                               tf.keras.metrics.FalsePositives(name='fp'),
                               tf.keras.metrics.TrueNegatives(name='tn'),
                               tf.keras.metrics.FalseNegatives(name='fn'),
                               tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc'),
                           ])
        self.model.fit(x=X_train, y=np.array([1 if item else 0 for item in y_train]), epochs=5,
                       validation_data=(X_val, np.array([1 if item else 0 for item in y_val])), validation_steps=10)

    def predict(self, reviews_test):
        return [prediction[0] for prediction in self.model.predict(self.encoder.transform(reviews_test))]
