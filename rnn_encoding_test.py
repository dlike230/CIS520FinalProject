from rnn import RNNEncoder
import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(3, 2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2)),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
data = RNNEncoder().fit_transform(["b c a", "a a b c"], np.array([0, 1]))
print(data)
model.fit(data, epochs=10)
