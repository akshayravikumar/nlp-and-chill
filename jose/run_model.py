import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from metrics import Measure


def run_epoch(data,val_set,test_set, model, optimizer, args, is_training):
    data_loader = torch.utils.data.DataLoader(data, batch_size = args.batch_size, shuffle = True)
    losses = 0
    if is_training:
        model.train()
    else:
        model.eval()
    count = 0
    for batch in tqdm(data_loader):
        print "%%%%%%%%%%%%%%%%%%%%%%%%"
        print count
        count += 1
        x = autograd.Variable(batch["x"])
        y = autograd.Variable(torch.zeros(batch["x"].shape[0])).long()
	
	if args.cuda:
		x=x.cuda()
		y=y.cuda()

        pad_title=batch["pad_title"]
        pad_body=batch["pad_body"]
        if is_training:
            optimizer.zero_grad()
        out = model(x,pad_title,pad_body)
        loss = F.multi_margin_loss(out, y, margin = 0.2)
        if is_training:
            loss.backward()
            optimizer.step()
        if count%15==1:
            dev_loader=torch.utils.data.DataLoader(val_set, batch_size = 1, shuffle = True)
            test_loader=torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False)
            dev_measure = Measure()
            test_measure=Measure()
	    batch_row=0
            for dev_batch in dev_loader:
                dev_x=autograd.Variable(dev_batch["x"])
                dev_pad_title=dev_batch["pad_title"]
                dev_pad_body=dev_batch["pad_body"]
		if args.cuda:
			dev_x=dev_x.cuda()
                out_dev_x=model(dev_x,dev_pad_title,dev_pad_body).data

                #out_dev_x=dev_batch["BM25"][0,:]
                ids=dev_batch["ids"][0,:]
                pos_ids=val_set.positives(batch_row)
                dev_measure.add_sample(out_dev_x.cpu(),None,ids[1:],pos_ids)
                batch_row+=1
	    batch_row=0
            for test_batch in test_loader:
                test_x=autograd.Variable(test_batch["x"])
		test_pad_title=test_batch["pad_title"]
		test_pad_body=test_batch["pad_body"]
		if args.cuda:
			test_x=test_x.cuda()
		out_test_x=model(test_x,test_pad_title,test_pad_body,).data
		ids=test_batch["ids"][0,:]
		pos_ids=test_set.positives(batch_row)
		test_measure.add_sample(out_test_x.cpu(),None,ids[1:],pos_ids)
		batch_row+=1
	    print ("DEEEEEEEEEEEEEEEEEEEEEEEV")
            print("MAP", str(dev_measure.MAP()))
            print("MRR", str(dev_measure.MRR()))
            print("P1",str(dev_measure.P1()))
            print("P5",str(dev_measure.P5()))
            print "Teeeeeeeeeeeeeeeeeeeest Loss"
	    print ("MAP", str(test_measure.MAP()))
	    print ("MRR", str(test_measure.MRR()))
	    print ("P1", str(test_measure.P1()))
	    print ("P5", str(test_measure.P5()))
        print("LOSS", loss.data[0])
        losses += loss.data
    return 1.0 * losses/count

def train_model(train_data, dev_data, test_data, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    for epoch in range(0, args.epochs):
        print("epoch: " + str(epoch))
        loss = run_epoch(train_data,dev_data,test_data, model, optimizer, args, True)

