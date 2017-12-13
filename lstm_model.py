import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from parse_utils import *

# CONV1D: in_channels = encoding dimension,
class LSTM(nn.Module):
    def __init__(self, hidden_size,input_size1,num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size=input_size1,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        #self.lstm2=nn.LSTM(input_size=input_size2,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cos = nn.CosineSimilarity()
        self.tanh = nn.Tanh()

    def forward(self, x,pad_title,pad_body,h):
        matrix = None
        for count, sample in enumerate(x):
            temp1=autograd.Variable(torch.diag(pad_title[count,:]).float())
            temp2=autograd.Variable(torch.diag(pad_body[count,:]).float())
            title = sample[:,:,:25].permute(0,2,1)
            body = sample[:,:,25:].permute(0,2,1)

            mesh1=np.repeat(np.invert(title.eq(0).data.numpy().all(2,keepdims=True)),self.hidden_size,axis=2)
            mesh1=mesh1.astype("int")
            mesh1=autograd.Variable(torch.from_numpy(mesh1).float())
            mesh2=np.repeat(np.invert(body.eq(0).data.numpy().all(2,keepdims=True)),self.hidden_size,axis=2)
            mesh2=mesh2.astype("int")
            mesh2=autograd.Variable(torch.from_numpy(mesh2).float())

            out1= self.lstm1(title,h)
            out1=mesh1*out1[0]
            out1=out1.permute(0,2,1)
            out1 = self.pool(out1)
            out1=F.linear(out1.permute(1,2,0)*25,temp1)
            out1=out1.permute(2,0,1)

            out2=self.lstm1(body,h)
            out2=mesh2*out2[0]
            out2=out2.permute(0,2,1)
            out2 = self.pool(out2)
            out2 = F.linear(out2.permute(1,2,0) * 100, temp2)
            out2=out2.permute(2,0,1)

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
    def init_hidden(self,batch_size):
        h0=autograd.Variable(torch.zeros(1,batch_size,self.hidden_size))
        c0=autograd.Variable(torch.zeros(1,batch_size,self.hidden_size))
        return h0,c0
