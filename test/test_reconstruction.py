from pipeline.fetch_data import fetch_data
from models.AutoEncoder import BagOfWordsAutoEncoder

text_data, df = fetch_data()
n = len(text_data)
train_data = text_data[:n//2]
test_data = text_data[n//2:]

autoencoder = BagOfWordsAutoEncoder()
autoencoder.fit_transform(train_data)
print(autoencoder.reconstruction_error(test_data))