import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "../datasets/"))
from mass_spring import get_dataset
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torch.utils.data as data
from TorchSnippet.energy import HNN
from TorchSnippet.dyna import odeint
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

t = torch.linspace(0,1, 100).reshape(-1,1)
X = torch.cat([
    torch.sin(2*np.pi*t),
    torch.cos(2*np.pi*t)
],1).to(device)

y = torch.cat([
    torch.cos(2*np.pi*t),
    -torch.sin(2*np.pi*t)
],1).to(device)
# data1 = get_dataset(samples=50)
# X, y = torch.tensor(data1['x'], dtype=torch.float32).to(device), torch.tensor(data1['dx'], dtype=torch.float32).to(device)
# print(X.shape, y.shape)
train = data.TensorDataset(X, y)
dataloader = data.DataLoader(train, batch_size=15, shuffle=False)


class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
        self.c = 0

    def forward(self, x):
        return self.model(x)

    def loss(self, y, y_hat):
        return ((y-y_hat)**2).sum()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return dataloader


func = HNN(nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Linear(64,1))).to(device)

learner = Learner(func)
trainer = pl.Trainer(min_epochs=500, max_epochs=1000)
trainer.fit(learner)

func1 = lambda t, x: func(x)

# x_t = torch.randn(1000, 2).to(device)
x_t = X[:1, :]
print(x_t.shape)
s_span = torch.linspace(0, 2*np.pi, 618)
trajectory = odeint(func1, x_t, s_span).detach().cpu().numpy()
for i in range(len(x_t)):
    plt.plot(trajectory[:, i, 0], trajectory[:, i, 1], 'b')
plt.plot(X[:, 0], X[:, 1], '+')
plt.show()
