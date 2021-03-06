import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from parse_utils import *
import random
import time

class AndroidData(data.Dataset):
    # X comes in w/ shape (a, b, c), a=# num_samples, b = 0 main query, b = 1 q+, b = 2 Q-. c is set to 20
    # pos_map --> {2: [3]}
    # neg_map {2: [4, 5, 6]}
    def __init__(self, pos_map=None, neg_map=None, size=None):
        if pos_map is None or neg_map is None:
            if size is None:
                raise Exception("If either pos/neg are None, size must not be None")
        if pos_map is None or neg_map is None:
            self.keys = list(id_to_title_target.keys())
        else:
            self.keys = list(pos_map.keys())

        self.size = size
        self.pos_map = pos_map
        self.neg_map = neg_map

    def __len__(self):
        return len(self.keys) if self.size is None else len(self.keys)/3

    def __getitem__(self, index):
        # Assumptions titles aren't longer than 25
        # EMBEDDING_DIM = encoding dim, 25 = max title length, 100 = max body length
        start_time = time.time()

        main_query = self.keys[index]
        main_title = question_to_vec(main_query, id_to_title_target)
        main_body = question_to_vec(main_query, id_to_body_target)
        # Some words have empty mappings, hence this hack just bypasses them
        if main_title is None or main_body is None:
            rand_ind = random.randint(0, len(self.keys) - 1)
            return self.__getitem__(rand_ind)
     
        title_matrix, comp1 = pad(torch.zeros(1, EMBEDDING_DIM, 25), main_title)
        body_matrix, comp2 = pad(torch.zeros(1, EMBEDDING_DIM, 100), main_body)

        comp_title_arr = [1.0 / comp1[0]]
        comp_body_arr = [1.0 / comp2[0]]

        if self.pos_map is None or self.neg_map is None:
            candidates = []
            truth = None
        else:
            pos = self.pos_map[main_query]
            neg = self.neg_map[main_query]
            candidates = pos + neg

        if len(candidates) == 0:
            title_matrix = torch.cat((title_matrix, torch.zeros(self.size - 1, EMBEDDING_DIM, 25)), 0)
            body_matrix = torch.cat((body_matrix, torch.zeros(self.size - 1, EMBEDDING_DIM, 100)), 0)
            comp_title_arr.extend([1]*(self.size - 1))
            comp_body_arr.extend([1]*(self.size - 1))       
        
        middle_time = time.time() 
        # print("middle time", middle_time - start_time)

        for candidate_index in candidates:
            candidate_title = question_to_vec(candidate_index, id_to_title_target)
            candidate_body = question_to_vec(candidate_index, id_to_body_target)

            if candidate_title is None or candidate_body is None:
                print("random stuff 2")
                rand_ind = random.randint(0, len(self.keys) - 1)
                return self.__getitem__(rand_ind)

            cand_title, comp_title = pad(torch.zeros(1, EMBEDDING_DIM, 25), candidate_title)
            cand_body, comp_body = pad(torch.zeros(1, EMBEDDING_DIM, 100), candidate_body)

            title_matrix = torch.cat((title_matrix, cand_title), 0)
            body_matrix = torch.cat((body_matrix, cand_body), 0)

            comp_title_arr.append(1.0 / comp_title[0])
            comp_body_arr.append(1.0 / comp_body[0])

        end_time = time.time()
        # print("end time", end_time - middle_time)

        comp_title_arr = np.asarray(comp_title_arr, dtype="double")
        comp_body_arr = np.asarray(comp_body_arr, dtype="double")
        # Concat the title and the bodies: we get 22xEMBEDDING_DIMx125. 125 = 25+100 (title+body)
        # 22 comes from: 1(main query)+1(positive_query)+20(negative samples)
        x_matrix = torch.cat((title_matrix, body_matrix), 2)

        # print("final time", time.time() - end_time)
        # print("TOTAL TIME", time.time() - start_time) 
        return {
            "x": x_matrix,
            "pad_title": torch.from_numpy(comp_title_arr),
            "pad_body": torch.from_numpy(comp_body_arr),
        }
