import tensorflow as tf

from models.Rnn import WordEncoder


class LstmAutoEncoder:

    def __init__(self, n_dimensions=32):
        self.text_seq_encoder = WordEncoder()
        self.model = None
        self.n_dimensions = n_dimensions

    def fit_transform(self, texts):
        encoded_sequences = self.text_seq_encoder.fit_transform(texts)
        encoded_sequences = encoded_sequences.reshape((len(texts), self.text_seq_encoder.padded_length))
        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.Embedding(input_dim=self.text_seq_encoder.padded_length, output_dim=64),
             tf.keras.layers.LSTM(100, activation='relu'),
             tf.keras.layers.Dense(self.n_dimensions, activation="softmax")])
        self.decoder = tf.keras.Sequential([tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.n_dimensions)),
                                            tf.keras.layers.RepeatVector(self.text_seq_encoder.padded_length),
                                            tf.keras.layers.LSTM(100, activation='relu', return_sequences=True)])
        self.model = tf.keras.Sequential(self.encoder, self.decoder)
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(x=encoded_sequences, y=encoded_sequences)
        return self._transform(encoded_sequences)

    def _transform(self, encoded_sequences):
        return self.model.predict(encoded_sequences)

    def transform(self, texts):
        return self._transform(
            self.text_seq_encoder.transform(texts).reshape(len(texts), self.text_seq_encoder.padded_length))
