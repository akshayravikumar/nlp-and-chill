import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np


# CONV1D: in_channels = encoding dimension,
class CNN(nn.Module):
    def __init__(self, hidden_size, window):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.window = window
        self.conv = nn.Sequential(nn.Conv1d(
            in_channels = 200,
            out_channels = hidden_size,
            kernel_size = window,
            stride = 1,
        ),
        nn.Tanh())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cos = nn.CosineSimilarity()

    def forward(self, x):
        matrix = None
        for count, sample in enumerate(x):
            title = sample[:,:,:25]
            body = sample[:,:,25:]

            out1 = self.conv(title)
            print out1.data[:2,0,:]
            out1 = self.pool(out1)

            out2 = self.conv(body)
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
