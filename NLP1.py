import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import argparse


#Returns an array of vecs for words and their mappings (word to index in array)
def embeddings(directory):
    mapping={}
    file=open(directory)
    embeddings = []
    count=0
    elt=file.readline()
    while elt !="":
        if count%10000==0:
            print count
        count+=1
        arr=elt.split()
        word,vector=arr[0],arr[1:]
        mapping[word]=count
        temp=[float(x) for x in vector]
        embeddings.append(temp)

        elt = file.readline()
    return embeddings,mapping
embed,map=embeddings("askubuntu-master/vectors_pruned_by_tests_fixed.200.txt")

def parse_file(file,(a,b,c)):
    matrix=np.zeros((a,b,c))

    query=file.readline()
    counter=0
    while query!="":
        arr=query.split("\t")
        if len(arr)==4:
            q,q_pos,Q_neg,BM=arr
            BM=BM.split()
            BM=[float(x) for x in BM]
            matrix[counter,3,:]=BM[:20]
        else:
            q,q_pos,Q_neg=arr
        q=int(q)
        q_pos=q_pos.split()
        q_pos=[int(x) for x in q_pos]
        Q_neg=Q_neg.split()
        Q_neg=[int(x) for x in Q_neg]

        q_pad=np.zeros((20,),dtype="int")-1
        q_pad[0]=q

        q_pos_pad=np.zeros((20,),dtype="int")-1
        q_pos_pad[:min(len(q_pos),20)]=q_pos[:min(20,len(q_pos))]

        q_neg_pad=Q_neg[:20]

        matrix[counter, 0, :] = q_pad
        matrix[counter, 1, :] = q_pos_pad
        matrix[counter, 2, :] = q_neg_pad

        query=file.readline()
        counter += 1
    return matrix
#Assuming num_negs>num_pos
num_pos=6
num_negs=20
def make_sets(dir_train,dir_test,dir_dev):
    train_file=open(dir_train)
    dev_file=open(dir_dev)
    test_file=open(dir_test)

    train=parse_file(train_file,(12724,3,num_negs))
    dev=parse_file(dev_file,(200,4,num_negs))
    test=parse_file(test_file,(200,4,num_negs))

    return train,dev,test
train,dev,test=make_sets("askubuntu-master/train_random.txt","askubuntu-master/dev.txt","askubuntu-master/test.txt")

def corpus(directory):
    id_to_title={}
    id_to_body={}
    file=open(directory)
    temp=file.readline()

    while temp!="":
        temp=temp.split("\t")
        id,title,body=temp
        id=int(id)

        title=title.split()
        body=body.split()

        id_to_title[id]=title
        id_to_body[id]=body

        temp=file.readline()
    return id_to_title,id_to_body
id_to_title,id_to_body=corpus("askubuntu-master/texts_raw_fixed.txt")

#Question id, mapping: either id_to_title or id_to_body
def question_to_vec(question,mapping):
    sentence=mapping.get(question)
    matrix=[]
    for word in sentence:
        if len(matrix)>100-1:
            break
        if word in map:
            id=map.get(word)
            vec=embed[id]
            matrix.append(vec)
    a=np.asarray(matrix).T

    x,y=a.shape
    a=a.reshape(1,x,y)
    garbage=torch.from_numpy(a).float()
    return garbage

class Data(data.Dataset):
    #X comes in w/ shape (a,b,c), a=#num_samples, b=0 main query, b=1 q+, b=2 Q-. c is set to 20
    def __init__(self,X):

        #Final shape: (queries*positive,1+|Q-|)
        self.Xmatrix=[]
        a,b,c=X.shape

        for eltx in range(a):
            mainQ=X[eltx,0,0]
            for elty in range(c):
                if X[eltx,1,elty]>0:
                    entry=[mainQ]
                    entry.append(X[eltx,1,elty])
                    for eltz in range(c):
                        entry.append(X[eltx,2,eltz])
                    self.Xmatrix.append(entry)
                else:
                    break
        self.Xmatrix=np.asarray(self.Xmatrix)
    def __len__(self):
        return len(self.Xmatrix)
    def __getitem__(self,index):
        #Assumptions titles aren't longer than 25
        garbage1=torch.zeros(1,200,25)
        garbage2=torch.zeros(1,200,100)
        main_query=self.Xmatrix[index,0]

        main_title=question_to_vec(main_query,id_to_title)
        main_body=question_to_vec(main_query,id_to_body)
        shape1=main_title.shape[-1]
        shape2=main_body.shape[-1]
        garbage1[0,:,:min(25,shape1)]=main_title[0,:,:min(25,shape1)]
        garbage2[0,:,:min(100,shape2)]=main_body[0,:,:min(100,shape2)]

        for elt in self.Xmatrix[index,1:]:
            query_title=question_to_vec(elt,id_to_title)
            query_body=question_to_vec(elt,id_to_body)
            garbo1=torch.zeros(1,200,25)
            garbo2=torch.zeros(1,200,100)
            shape1=query_title.shape[-1]
            shape2=query_body.shape[-1]


            garbo1[0,:,:min(25,shape1)]=query_title[0,:,:min(25,shape1)]
            garbo2[0,:,:min(100,shape2)]=query_body[0,:,:min(100,shape2)]

            garbage1=torch.cat((garbage1,garbo1),0)
            garbage2=torch.cat((garbage2,garbo2),0)
        temp=torch.cat((garbage1,garbage2),2)
        return {"x":torch.cat((garbage1,garbage2),2)}
training=Data(train)



class CNN(nn.Module):
    def __init__(self,hidden_size,window):
        super(CNN,self).__init__()
        self.hidden_size=hidden_size
        self.window=window
        self.conv=nn.Sequential(nn.Conv1d(
            in_channels=200,
            out_channels=hidden_size,
            kernel_size=3,
            stride=1,
        ),
        nn.Tanh())
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.cos=nn.CosineSimilarity()
    def forward(self,x):
        matrix=None
        for count,sample in enumerate(x):
            title=sample[:,:,:25]
            body=sample[:,:,25:]

            out1=self.conv(title)
            out1=self.pool(out1)

            out2=self.conv(body)
            out2=self.pool(out2)

            out=self.pool(torch.cat((out1,out2),2))

            rows=out.data.shape[0]
            main=out[0,:,0].repeat(rows-1,1)
            Q=out[1:,:,0]

            final_out=self.cos(Q,main)
            if count==0:
                matrix=final_out
            else:
                matrix=torch.cat((matrix,final_out))
        return matrix

def parser():
    parser=argparse.ArgumentParser(description="NLP project P1")
    parser.add_argument("--lr",type=float, default=0.1)
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--weight",type=float,default=1e-3)
    return parser.parse_args()
args=parser()
def run_epoch(data,model,optimizer,args,is_training):
    data_loader=torch.utils.data.DataLoader(data,batch_size=args.batch_size,shuffle=False)
    losses=[]

    if is_training:
        model.train()
    else:
        model.eval()
    count=0
    for batch in tqdm(data_loader):
        print count
        count+=1
        x=autograd.Variable(batch["x"])
        y=autograd.Variable(torch.zeros(args.batch_size))

        if is_training:
            optimizer.zero_grad()
        out=model(x)

        loss=F.multi_margin_loss(out,y.long())

        if is_training:
            loss.backward()
            optimizer.step()
        losses.append(loss)
    return np.mean(losses)
def train_model(train_data,dev_data,test_data,model):
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight)

    for epoch in range(0,args.epochs):
        print "epoch: "+str(epoch)
        loss=run_epoch(train_data,model,optimizer,args,True)
net=CNN(500,3)
results=train_model(training,None,None,net)





















