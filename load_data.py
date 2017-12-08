import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from parse_utils import *

class Data(data.Dataset):
    # X comes in w/ shape (a, b, c), a=# num_samples, b = 0 main query, b = 1 q+, b = 2 Q-. c is set to 20
    def __init__(self, X):

        # Final shape: (queries*positive,1+|Q-|)
        self.Xmatrix = []
        a, b, c = X.shape

        for eltx in range(a):
            mainQ = X[eltx, 0, 0]
            for elty in range(c):
                if X[eltx, 1, elty]>0:
                    entry = [mainQ]
                    entry.append(X[eltx,1, elty])
                    for eltz in range(c):
                        entry.append(X[eltx,2, eltz])
                    self.Xmatrix.append(entry)
                else:
                    break
        self.Xmatrix = np.asarray(self.Xmatrix)
    def __len__(self):
        return len(self.Xmatrix)
    def __getitem__(self, index):
        # Assumptions titles aren't longer than 25
        # 200 = encoding dim, 25 = max title length, 100 = max body length
        main_query = self.Xmatrix[index, 0]
        main_title = question_to_vec(main_query, id_to_title)
        main_body = question_to_vec(main_query, id_to_body)
        # Some words have empty mappings, hence this hack just bypasses them
        if main_title is None or main_body is None:
            return {"x" : torch.zeros(22, 200, 125)}
        garbage1=pad(torch.zeros(1, 200, 25),main_title)
        garbage2=pad(torch.zeros(1,200,100),main_body)
        for elt in self.Xmatrix[index,1:]:
            query_title = question_to_vec(elt, id_to_title)
            query_body = question_to_vec(elt, id_to_body)
            #  Some words have empty mappings, hence this hack just bypasses them
            if query_title is None or query_body is None:
                return {"x" : torch.zeros(22, 200, 125)}
            garbo1=pad(torch.zeros(1, 200, 25),query_title)
            garbo2=pad(torch.zeros(1, 200, 100),query_body)
            garbage1 = torch.cat((garbage1, garbo1),0)
            garbage2 = torch.cat((garbage2, garbo2),0)
        # Concat the title and the bodies: we get 22x200x125. 125 = 25+100 (title+body)
        # 22 comes from: 1(main query)+1(positive_query)+20(negative samples)
        return {"x":torch.cat((garbage1, garbage2),2)}
class Eval_Data(data.Dataset):
    #Expecting data to be of dims: (around 200,4,#negs)
    def __init__(self,data):
        self.data=data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,i):
        truth=torch.zeros(1)
        lie=torch.ones(1)
        BM25=torch.from_numpy(self.data[i,3,:])
        ids=np.zeros(21)
        ids[0]=self.data[i,0,0]
        ids[1:]=self.data[i,2,:]
        ids=torch.from_numpy(ids)
        positive_ids=[]
        #Get none padded positive ids
        for elt in self.data[i,1,:]:
            if elt>=0:
                positive_ids.append(elt)
        positive_ids=torch.from_numpy(np.asarray(positive_ids))
        mainQ=self.data[i,0,0]
        mainQ_title=question_to_vec(mainQ,id_to_title)
        mainQ_body=question_to_vec(mainQ,id_to_body)
        if mainQ_title is None or mainQ_body is None:
            return {"x" : torch.zeros(21, 200, 125),"BM25":BM25,
                    "ids":ids,"good":lie}
        garbage1=pad(torch.zeros(1,200,25),mainQ_title)
        garbage2=pad(torch.zeros(1,200,100),mainQ_body)
        for elt in self.data[i,2,:]:
            query_title=question_to_vec(elt,id_to_title)
            query_body=question_to_vec(elt,id_to_body)
            if query_title is None or query_body is None:
                return {"x": torch.zeros(21, 200, 125),"BM25":BM25,
                        "ids":ids,"good":lie}
            garbo1=pad(torch.zeros(1,200,25),query_title)
            garbo2=pad(torch.zeros(1,200,100),query_body)
            garbage1=torch.cat((garbage1,garbo1),0)
            garbage2=torch.cat((garbage2,garbo2),0)
        return {"x":torch.cat((garbage1,garbage2),2),"BM25":BM25,
                "ids":ids,"good":truth}
    def positives(self,i):
        #Lookup positive ids for a given id query
        positive_ids=[]
        for elt in self.data[i,1,:]:
            if elt>=0:
                positive_ids.append(elt)
        return positive_ids



