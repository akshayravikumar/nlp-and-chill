import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np


# CONV1D: in_channels = encoding dimension,
class DomainClassifier(nn.Module):
    def __init__(self, D_in, args):
        super(DomainClassifier, self).__init__()
        self.args = args 
        self.D_in = D_in
        self.D_h1 = 300
        self.D_h2 = 150
        self.D_out = 2
        self.model = nn.Sequential(
            nn.Linear(self.D_in, self.D_h1),
#             nn.ReLU(),
            nn.Linear(self.D_h1, self.D_h2),
            nn.ReLU(),
            nn.Linear(self.D_h2, self.D_out),
        )

    def forward(self, x):
        out = self.model(x)
        # out = F.softmax(out)
        if self.args.cuda:
            out = out.cuda()
        return out
