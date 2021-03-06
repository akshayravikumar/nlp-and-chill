import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from parse_utils import *
import random

class Data(data.Dataset):
    # X comes in w/ shape (a, b, c), a=# num_samples, b = 0 main query, b = 1 q+, b = 2 Q-. c is set to 20
    def __init__(self, X):

        # Final shape: (queries*positive, 1+|Q-|)
        self.Xmatrix = []
        a, b, c = X.shape

        for eltx in range(a):
            mainQ = X[eltx, 0, 0]
            for elty in range(c):
                if X[eltx, 1, elty]>0:
                    entry = [mainQ]
                    entry.append(X[eltx, 1, elty])
                    for eltz in range(c):
                        entry.append(X[eltx, 2, eltz])
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
            print "lol1"
            rand_ind= random.randint(0,len(self.Xmatrix)-1)
            return self.__getitem__(rand_ind)
            # print "lol1"
            # return {"x": torch.zeros(22, 200, 125),"pad_title":torch.from_numpy(np.zeros(22,dtype="double")),
            #             "pad_body":torch.from_numpy(np.zeros(22,dtype="double"))}
        garbage1,comp1=pad(torch.zeros(1, 200, 25),main_title)
        garbage2,comp2=pad(torch.zeros(1,200,100),main_body)
        comp1=[1.0/comp1[0]]
        comp2=[1.0/comp2[0]]
        for elt in self.Xmatrix[index, 1:]:
            query_title = question_to_vec(elt, id_to_title)
            query_body = question_to_vec(elt, id_to_body)
            #  Some words have empty mappings, hence this hack just bypasses them
            if query_title is None or query_body is None:
                print "lol2"
                rand_ind = random.randint(0, len(self.Xmatrix) - 1)
                return self.__getitem__(rand_ind)
                # print "lol2"
                # return {"x" : torch.zeros(22, 200, 125),"pad_title":torch.from_numpy(np.zeros(22,dtype="double")),
                #         "pad_body":torch.from_numpy(np.zeros(22,dtype="double"))}
            garbo1,comp3 = pad(torch.zeros(1, 200, 25), query_title)
            garbo2,comp4 = pad(torch.zeros(1, 200, 100), query_body)
            garbage1 = torch.cat((garbage1, garbo1), 0)
            garbage2 = torch.cat((garbage2, garbo2), 0)
            comp1.append(1.0/comp3[0])
            comp2.append(1.0/comp4[0])
        comp1=np.asarray(comp1,dtype="double")
        comp2=np.asarray(comp2,dtype="double")
        # Concat the title and the bodies: we get 22x200x125. 125 = 25+100 (title+body)
        # 22 comes from: 1(main query)+1(positive_query)+20(negative samples)
        return {"x":torch.cat((garbage1, garbage2), 2),"pad_title":torch.from_numpy(comp1),
                "pad_body":torch.from_numpy(comp2)}

class Eval_Data(data.Dataset):
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
            print "lol1"
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
                print "lol2"
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

class TGIF(data.Dataset):
	#Need file as input
	def __init__(self,fil):
		pass
	def __len__(self):
		pass
	def __getitem__(self,i):
		pass

