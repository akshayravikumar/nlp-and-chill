import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np


class Measure(object):
    def __init__(self):
        self.score1 = []
        self.score2 = []
        self.score3 = []
        self.score4 = []

    def add_sample(self, out, BM25, ids, pos_ids):
        out = out[0, :]
        #Calculate MRR,MAP for each iteration
        #Map only for positive
        temp = []
        count2 = 0
        pos_indices = {}
        for index, gen_id in enumerate(ids):
            for pos_id in pos_ids:
                if int(pos_id) == int(gen_id):
                    pos_indices[index] = pos_id
        #Greatest to least
        count = 0.0
        sorted_indices = np.argsort(out.numpy())[::-1]
        easy = []
        for rank, x in enumerate(sorted_indices):
            #Is x a positive index
            if x in pos_indices:
                #For MRR
                if count == 0.0:
                    self.score2.append(1.0 / (rank + 1))
                #P@1
                if rank == 0:
                    self.score3.append(1.0)

                #P@5
                if rank < 5:
                    count2 += 1
                count += 1.0
                temp.append(count / (rank + 1))
        self.score4.append(1.0 * count2 / 5.0)
        self.score1.append(
            sum(temp) /
            len(temp)) if len(temp) > 0 else self.score1.append(0.0)

    def MRR(self):
        return 1.0 * sum(self.score2) / len(self.score1)

    def MAP(self):
        return 1.0 * sum(self.score1) / len(self.score1)

    def P1(self):
        return 1.0 * sum(self.score3) / len(self.score1)

    def P5(self):
        return 1.0 * sum(self.score4) / len(self.score1)


#Need to check BM25 rankings with these functions
