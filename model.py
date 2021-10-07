import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_dim):
        super(Model, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 500),
            nn.BatchNorm1d(500),
            nn.Sigmoid(),
            
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.Sigmoid(),

            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.Sigmoid(),

            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = self.net(x)

        return x