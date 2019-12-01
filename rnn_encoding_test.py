from rnn import RNNEncoder
import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5, 2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2)),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
encoder = RNNEncoder()
data = encoder.fit_transform(["b c a", "a a b c"])
print(data)
model.fit(x=data, y=np.array([0, 1]), epochs=10)
predictions = model.predict(x=encoder.transform(["a a b c", "g", "sdafasdfs", "a", "a", "asdfasfasda a a a"]))
print([round(prediction[0]) for prediction in predictions])
