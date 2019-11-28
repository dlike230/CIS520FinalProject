import tensorflow as tf
import nltk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.layers.embeddings import Embedding
from tensorflow_core.python.keras.layers.recurrent import LSTM
from tensorflow_core.python.keras.layers.wrappers import Bidirectional
from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
import tensorflow_datasets as tfds

model = Sequential([
    Embedding(10, 64),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer=Adam(1e-4),
              metrics=['accuracy'])
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
encoder = info.features['text'].encoder


train_dataset, test_dataset = dataset['train'], dataset['test']
