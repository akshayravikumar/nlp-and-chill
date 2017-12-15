import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from parse_utils import *
import random


class ShuffleData(data.Dataset):
    # X comes in w/ shape (a, b, c), a=# num_samples, b = 0 main query, b = 1 q+, b = 2 Q-. c is set to 20
    def __init__(self, *datasets):
        self.datasets = datasets
        self.arr = []
        for i in range(len(self.datasets)):
            for j in range(len(self.datasets[i])):
                self.arr.append((i, j))
        random.shuffle(self.arr)

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    def __getitem__(self, index):
        i, j = self.arr[index]
        entry = self.datasets[i][j]
        entry["dataset"] = i
        return entry
