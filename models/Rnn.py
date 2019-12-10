import os

import numpy as np
import tensorflow as tf

from text_encoders.CharacterEncoder import CharacterRNNEncoder
from text_encoders.WordEncoder import WordEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class RNN:

    def __init__(self, encode_words=True):
        if encode_words:
            self.encoder = WordEncoder()
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
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(1e-4),
                           metrics=[
                               tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc'),
                           ])
        self.model.fit(x=X_train, y=np.array([1 if item else 0 for item in y_train]), epochs=5,
                       validation_data=(X_val, np.array([1 if item else 0 for item in y_val])), validation_steps=10)

    def predict(self, reviews_test):
        return [prediction[0] for prediction in self.model.predict(self.encoder.transform(reviews_test))]
