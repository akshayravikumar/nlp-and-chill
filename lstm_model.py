import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from parse_utils import *

# CONV1D: in_channels = encoding dimension,
class LSTM(nn.Module):
    def __init__(self, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.Sequential(nn.LSTM(
            input_size = 100,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True
        ),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cos = nn.CosineSimilarity()
        self.tanh = nn.Tanh()

    def forward(self, x):
        matrix = None
        for count, sample in enumerate(x):
            title = sample[:,:,:25]
            title = pad(np.zeros((title.data.shape[0], title.data.shape[1], 100)), title.data.numpy())
            body = sample[:,:,25:]

            out1, _ = self.lstm(autograd.Variable(torch.from_numpy(title)).float())
            out1 = self.tanh(out1)
            out1 = self.pool(out1)

            out2, _ = self.lstm(body)
            out12= self.tanh(out2)
            out2 = self.pool(out2)

            out = self.pool(torch.cat((out1, out2),2))

            rows = out.data.shape[0]
            main = out[0,:,0].repeat(rows-1, 1)
            Q = out[1:,:,0]

            final_out = self.cos(Q, main)
            if count == 0:
                matrix = final_out
            else:
                matrix = torch.cat((matrix, final_out))
        return matrix
