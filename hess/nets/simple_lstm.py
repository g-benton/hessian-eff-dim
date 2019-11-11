import torch
import math
from torch import nn

class LSTM(nn.Module):

    def __init__(self, train_x, train_y, n_hidden=2,  hidden_size=10,
                 input_dim=1, output_dim=1,
                 batch_size=1):
        super(LSTM, self).__init__()
        self.input_dim = 1
        self.hidden_dim = hidden_size
        self.batch_size = batch_size
        self.num_layers = n_hidden

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, 1)

    def forward(self, input):
        out, _ = self.lstm(input)
        return self.linear(out)
