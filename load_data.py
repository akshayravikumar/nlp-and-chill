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
        garbage1 = torch.zeros(1, 200, 25)
        garbage2 = torch.zeros(1, 200, 100)
        main_query = self.Xmatrix[index, 0]

        main_title = question_to_vec(main_query, id_to_title)
        main_body = question_to_vec(main_query, id_to_body)
        # Some words have empty mappings, hence this hack just bypasses them
        if main_title is None or main_body is None:
            return {"x" : torch.zeros(22, 200, 125)}

        shape1 = main_title.shape[-1]
        shape2 = main_body.shape[-1]
        garbage1[0,:,:min(25, shape1)] = main_title[0,:,:min(25, shape1)]
        garbage2[0,:,:min(100, shape2)] = main_body[0,:,:min(100, shape2)]

        for elt in self.Xmatrix[index,1:]:
            query_title = question_to_vec(elt, id_to_title)
            query_body = question_to_vec(elt, id_to_body)
            #  Some words have empty mappings, hence this hack just bypasses them
            if query_title is None or query_body is None:
                return {"x" : torch.zeros(22, 200, 125)}
            garbo1 = torch.zeros(1, 200, 25)
            garbo2 = torch.zeros(1, 200, 100)
            shape1 = query_title.shape[-1]
            shape2 = query_body.shape[-1]

            garbo1[0,:,:min(25, shape1)] = query_title[0,:,:min(25, shape1)]
            garbo2[0,:,:min(100, shape2)] = query_body[0,:,:min(100, shape2)]

            garbage1 = torch.cat((garbage1, garbo1),0)
            garbage2 = torch.cat((garbage2, garbo2),0)
        # Concat the title and the bodies: we get 22x200x125. 125 = 25+100 (title+body)
        # 22 comes from: 1(main query)+1(positive_query)+20(negative samples)
        return {"x":torch.cat((garbage1, garbage2),2)}

