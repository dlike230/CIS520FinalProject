import tensorflow as tf


class WordEmbeddingModel:

    def fit(self, vectors, train_labels):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, input_dim=vectors.shape[1], activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
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
        self.model.fit(x=vectors, y=train_labels, epochs=2)

    def predict(self, vectors):
        return self.model.predict(vectors)

    def get_params(self, deep = True):
        return {}
