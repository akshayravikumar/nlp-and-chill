import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np

class Measure(object):
    def __init__(self):
        score1=[]
        score2=[]
    def add_sample(self,out,BM25,ids,pos_ids):
        #Calculate MRR,MAP for each iteration
        #Map only for positive
        pos_indices=[]
        for xindex,elt in enumerate(ids):
            for pos_id in pos_ids:
                if elt==pos_id:
                    pass


    def MRR(self):
        pass
    def MAP(self):
        pass
    def P(self,n):
        pass