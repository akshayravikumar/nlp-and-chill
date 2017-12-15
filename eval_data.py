import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from parse_utils import *
import random


class EvalData(data.Dataset):
    #Expecting data to be of dims: (around 200, 4,#negs)
    def __init__(self, data):
        self.data = data
        self.last_returned=-1
        #print self.data.shape
    def __len__(self):
        return self.data.shape[0]
        #print self.data.shape[0]
        #return 186
    def __getitem__(self, i):
        truth = torch.zeros(1)
        lie = torch.ones(1)
        BM25 = torch.from_numpy(self.data[i, 3,:])
        ids = np.zeros(21)
        ids[0] = self.data[i, 0, 0]
        ids[1:] = self.data[i, 2,:]
        ids = torch.from_numpy(ids)
        positive_ids = []
        #Get none padded positive ids
        for elt in self.data[i, 1,:]:
            if elt>=0:
                positive_ids.append(elt)
        positive_ids = torch.from_numpy(np.asarray(positive_ids))
        mainQ = self.data[i, 0, 0]
        mainQ_title = question_to_vec(mainQ, id_to_title)
        mainQ_body = question_to_vec(mainQ, id_to_body)
        if mainQ_title is None or mainQ_body is None:
            rand_ind= random.randint(0,len(self.data)-1)
            return self.__getitem__(rand_ind)
        garbage1,comp1 = pad(torch.zeros(1, 200, 25), mainQ_title)
        garbage2,comp2 = pad(torch.zeros(1, 200, 100), mainQ_body)
        comp1=[1.0/comp1[0]]
        comp2=[1.0/comp2[0]]
        for elt in self.data[i, 2, :]:
            query_title = question_to_vec(elt, id_to_title)
            query_body = question_to_vec(elt, id_to_body)
            if query_title is None or query_body is None:
                rand_ind = random.randint(0, len(self.data) - 1)
                return self.__getitem__(rand_ind)
            garbo1, comp3 = pad(torch.zeros(1, 200, 25), query_title)
            garbo2, comp4 = pad(torch.zeros(1, 200, 100), query_body)
            garbage1 = torch.cat((garbage1, garbo1), 0)
            garbage2 = torch.cat((garbage2, garbo2), 0)
            comp1.append(1.0/comp3[0])
            comp2.append(1.0/comp4[0])
        comp1=np.asarray(comp1,dtype="double")
        comp2=np.asarray(comp2,dtype="double")
        self.last_returned=i
        return {"x":torch.cat((garbage1, garbage2), 2),"BM25":BM25,
                "ids":ids,"good":truth,"pad_title":torch.from_numpy(comp1),
                "pad_body":torch.from_numpy(comp2)}

    def positives(self, i):
        if i!=self.last_returned:
            return self.positives(self.last_returned)
        #Lookup positive ids for a given id query
        positive_ids=[]
        for elt in self.data[i,1,:]:
            if elt>=0:
                positive_ids.append(int(elt))
        positive_ids=np.asarray(positive_ids,dtype="int")
        return torch.from_numpy(positive_ids)



