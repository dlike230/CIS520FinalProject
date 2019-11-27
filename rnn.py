import tensorflow as tf
import nltk
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.layers.embeddings import Embedding
from tensorflow_core.python.keras.layers.recurrent import LSTM
from tensorflow_core.python.keras.layers.wrappers import Bidirectional
from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.keras.optimizers import Adam


class RNNEncoder:

    def __init__(self):
        self.word_index = {}

    @staticmethod
    def tokenize(train_strings):
        return [[word for tokenized_sentence in
                 (nltk.word_tokenize(sentence) for sentence in nltk.sent_tokenize(train_string)) for word in
                 tokenized_sentence] for train_string in train_strings]

    def fit_transform(self, train_strings):
        tokenized = RNNEncoder.tokenize(train_strings)
        all_words = list({word for review in tokenized for word in review})
        for i, word in enumerate(all_words):
            self.word_index[word] = i
        return self._transform(tokenized)

    def _transform(self, tokenized):
        total_list = []
        for review in tokenized:
            current_list = []
            for word in review:
                current_list.append(self.word_index[word] if word in self.word_index else len(self.word_index))
            total_list.append(current_list)
        return total_list

    def transform(self, strings):
        return self.transform(self.tokenize(strings))

    @property
    def vocab_size(self):
        return len(self.word_index)


class RNN:

    def __init__(self):
        self.encoder = RNNEncoder()
        self.model = None

    def fit(self, train_strings, y_train):
        encoded = self.encoder.fit_transform(train_strings)
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
        self.model.fit(X=encoded, y=y_train, epochs=10)

    def predict(self, reviews_test):
        return self.model.predict(self.encoder.transform(reviews_test))
