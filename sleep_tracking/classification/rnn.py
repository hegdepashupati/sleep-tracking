import numpy as np
import torch
from torch import nn
import random

torch.manual_seed(123)
random.seed(123)
np.random.seed(123)


class ClassifierRNN(nn.Module):
    def __init__(self, num_features, num_latents, num_classes, num_layers, dropout=0.1):
        super(ClassifierRNN, self).__init__()
        dropout = dropout if num_layers > 1 else 0.
        self.rnn = nn.GRU(input_size=num_features, hidden_size=num_latents, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.emission = nn.Sequential(nn.Linear(num_latents, num_latents), nn.Tanh(),
                                      nn.Linear(num_latents, num_classes))
        self.h0 = nn.Parameter(torch.zeros((num_layers, 1, num_latents)))

    def forward(self, x):
        latents, _ = self.rnn(x, self.h0)
        out = self.emission(latents).permute(0, 2, 1)
        return out

    def predict_latents(self, x):
        with torch.no_grad():
            self.eval()
            latents, _ = self.rnn(x, self.h0)
        return latents

    def predict_proba(self, x):
        with torch.no_grad():
            self.eval()
            latents, _ = self.rnn(x, self.h0)
            out = self.emission(latents)
            out = torch.softmax(out, dim=2)
        return out

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            latents, _ = self.rnn(x, self.h0)
            out = self.emission(latents)
            out = out.max(2)[1]
        return out

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
