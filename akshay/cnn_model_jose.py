import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np


# CONV1D: in_channels = encoding dimension,
class CNN(nn.Module):
    def __init__(self, hidden_size, window,args):
        super(CNN, self).__init__()
	self.args=args
        self.hidden_size = hidden_size
        self.window = window
        self.conv = nn.Sequential(nn.Conv1d(
            in_channels = EMBEDDING_DIM,
            out_channels = hidden_size,
            kernel_size = window,
            stride = 1,bias=True
        ),
        nn.Tanh())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cos = nn.CosineSimilarity()

    def forward(self, x,pad_title,pad_body):
        matrix = None
        for count, sample in enumerate(x):
            temp1=autograd.Variable(torch.diag(pad_title[count,:]).float())
            temp2=autograd.Variable(torch.diag(pad_body[count,:]).float())
	    if self.args.cuda:
		temp1=temp1.cuda()
		temp2=temp2.cuda()
            title = sample[:,:,:25]
            body = sample[:,:,25:]
            #Find the number of none padded:

            out1 = self.conv(title)
            #print out1.data.shape
            out1 = self.pool(out1)
            #print out1.data.shape

            out1=F.linear(out1.permute(1,2,0)*23.0,temp1)
            out1=out1.permute(2,0,1)
            #print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55"
            out2 = self.conv(body)
            out2 = self.pool(out2)
            out2 = F.linear(out2.permute(1, 2, 0) * 98.0, temp2)
            out2=out2.permute(2,0,1)

            out = self.pool(torch.cat((out1, out2),2))

            rows = out.data.shape[0]
            main = out[0,:,0].repeat(rows-1, 1)
            Q = out[1:,:,0]
            if sample.data.shape[0]==22:
                final_out=autograd.Variable(torch.zeros(1,21)).cuda()
                final_out[0,:] = self.cos(Q, main)
            else:
                final_out=autograd.Variable(torch.zeros(1,20)).cuda()
                final_out[0,:] = self.cos(Q, main)
            if count == 0:
                matrix = final_out
            else:
                matrix = torch.cat((matrix, final_out),0)
        return matrix
