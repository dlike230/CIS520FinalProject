import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from torch import nn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class AutoencoderNN(nn.Module):
    def __init__(self, in_dim, n=64):
        super(AutoencoderNN, self).__init__()
        self.encoder = nn.Sequential(
                                     nn.Linear(in_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, n),
                                     nn.ReLU(),)
        self.decoder = nn.Sequential(nn.Linear(n, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, in_dim),
                                     nn.Tanh())

    def forward(self, x):
        model = nn.Sequential(self.encoder, self.decoder)
        return model(x)


class BagOfWordsAutoEncoder:

    def __init__(self, num_epochs=10):
        self.vectorizer = CountVectorizer()
        self.model = None
        self.num_epochs = num_epochs

    def fit_transform(self, reviews, validation_set=None):
        vectors = self.vectorizer.fit_transform(reviews).toarray()
        in_size = vectors.shape[1]
        self.model = AutoencoderNN(in_size, n=32)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1E-4)
        trainloader = DataLoader(vectors, batch_size=32, shuffle=True)
        validation_losses = []
        for epoch in range(self.num_epochs):
            total_loss = 0
            num_items = 0
            print("Epoch %d" % epoch)
            for i, data in enumerate(trainloader):
                data = data.float()
                optimizer.zero_grad()
                output = self.model.forward(data)
                loss = criterion(output, data)
                loss_val = loss.item()
                total_loss += loss_val
                num_items += 1
                loss.backward()
                optimizer.step()
                if i % 20 == 0 and validation_set is not None:
                    val_loss = self.reconstruction_error(validation_set)
                    print("Validation loss: %f" % val_loss)
                    validation_losses.append(val_loss)
        # plt.plot(batch_losses)
        # plt.show()
        return self._transform(vectors)

    def _transform(self, vectors):
        return self.model.encoder(torch.from_numpy(vectors).float()).detach().numpy()

    def reconstruction_error(self, reviews):
        vectors = self.vectorizer.transform(reviews).toarray().astype("float32")
        encoded = self._transform(vectors)
        decoded = self.model.decoder(torch.from_numpy(encoded)).detach().numpy()
        return np.linalg.norm(vectors - decoded) ** 2 / len(reviews)

    def transform(self, reviews):
        return self._transform(self.vectorizer.transform(reviews).toarray())
