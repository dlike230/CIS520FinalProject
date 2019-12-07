import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class AutoencoderNN(nn.Module):
    def __init__(self, in_dim, n=64):
        super(AutoencoderNN, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, n),
                                     nn.ReLU(),)
        self.decoder = nn.Sequential(nn.Linear(n, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, in_dim),
                                     nn.Tanh())

    def forward(self, x):
        model = nn.Sequential(self.encoder, self.decoder)
        return model(x)


class BagOfWordsAutoEncoder:

    def __init__(self, num_epochs=3):
        self.vectorizer = TfidfVectorizer()
        self.encoder = None
        self.decoder = None
        self.model = None
        self.num_epochs = num_epochs

    def fit_transform(self, reviews):
        vectors = self.vectorizer.fit_transform(reviews).toarray()
        in_size = vectors.shape[1]
        self.model = AutoencoderNN(in_size, n=2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1E-4)
        trainloader = DataLoader(vectors, batch_size=32, shuffle=True)
        batch_losses = []
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
                batch_losses.append(loss_val)
                total_loss += loss_val
                num_items += 1
                loss.backward()
                optimizer.step()
                if i % 20 == 0:
                    print("Running loss estimate: %f" % (total_loss / num_items))
        plt.plot(batch_losses)
        plt.show()
        return self._transform(vectors)

    def _transform(self, vectors):
        return self.model.encoder(torch.from_numpy(vectors).float()).detach().numpy()

    def reconstruction_error(self, reviews):
        vectors = self.vectorizer.transform(reviews).toarray()
        encoded = self._transform(vectors)
        decoded = np.array(self.decoder.predict(encoded))
        print(decoded.shape)
        print(vectors.shape)
        return sum(np.linalg.norm(decoded_vec, vector) for vector, decoded_vec in zip(vectors, decoded)) / len(vectors)

    def transform(self, reviews):
        return self._transform(self.vectorizer.transform(reviews).toarray())
