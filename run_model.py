import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from metrics import Measure


def run_epoch(data,val_set,test_set, model, optimizer, args, is_training):
    data_loader = torch.utils.data.DataLoader(data, batch_size = args.batch_size, shuffle = False)

    losses = 0
    if is_training:
        model.train()
    else:
        model.eval()
    count = 0
    for batch in tqdm(data_loader):
        count += 1
        x = autograd.Variable(batch["x"])
        y = autograd.Variable(torch.zeros(args.batch_size))
        if is_training:
            optimizer.zero_grad()
        out = model(x)
        loss = F.multi_margin_loss(out, y.long(), margin = 0.2)
        if is_training:
            loss.backward()
            optimizer.step()
        if count%10==1:
            dev_loader=torch.utils.data.DataLoader(val_set, batch_size = 1, shuffle = False)
            #test_loader=torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False)
            dev_measure = Measure()
            batch_row=0
            for dev_batch in dev_loader:
                dev_x=autograd.Variable(dev_batch["x"])
                out_dev_x=model(dev_x).data
                ids=dev_batch["ids"][0,:]
                pos_ids=val_set.positives(batch_row)
                dev_measure.add_sample(out_dev_x,None,ids[1:],pos_ids)
                batch_row+=1
            # for test_batch in test_loader:
            #     test_x=autograd.Variable(test_batch["x"])
            #     BM25=dev_batch["BM25"]
            #     ids=dev_batch["id"]
            print("MAP", str(dev_measure.MAP()))
            print("MRR", str(dev_measure.MRR()))
            #print "Test Loss"
        print("LOSS", loss.data[0])
        losses += loss.data
    return 1.0 * losses/count

def train_model(train_data, dev_data, test_data, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    for epoch in range(0, args.epochs):
        print("epoch: " + str(epoch))
        loss = run_epoch(train_data,dev_data,test_data, model, optimizer, args, True)

