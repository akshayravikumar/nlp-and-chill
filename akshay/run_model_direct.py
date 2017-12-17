import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from metrics import Measure
from meter import AUCMeter
import time

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == autograd.Variable:
        return autograd.Variable(h.data).cuda()
    else:
        return tuple(repackage_hidden(v) for v in h)

def run_epoch(train_data, dev_data, test_data, model, domain_model, 
                optimizer_feature, optimizer_domain, args, is_training):
    data_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)
    losses_feature = 0
    losses_domain = 0
    if is_training:
        model.train()
    else:
        model.eval()

    dev_batch_size = 15
    count = 0
    def evaluate(dataset):
        dev_loader = torch.utils.data.DataLoader(dataset, batch_size=dev_batch_size, shuffle=True)
        start_time = time.time()
        for dev_batch in tqdm(dev_loader):
            print("TIME START", time.time() - start_time)

            dev_x = autograd.Variable(dev_batch["x"])
            if args.cuda:
                dev_x = dev_x.cuda()

            dev_pad_title = dev_batch["pad_title"]
            dev_pad_body = dev_batch["pad_body"]
    
            if args.model == "lstm":
                hidden2 = model.init_hidden(dev_x.shape[1])
                hidden2 = repackage_hidden(hidden2)
                out_dev_x_raw, _ = model(dev_x, dev_pad_title, dev_pad_body, hidden2)
            else:
                out_dev_x_raw, _ = model(dev_x, dev_pad_title, dev_pad_body)

            # out_dev_x_raw, _ = model(dev_x, dev_pad_title, dev_pad_body)
            out_dev_x = out_dev_x_raw.data

            truth = [0] * len(out_dev_x[0])
            truth[0] = 1
            truth = np.asarray(truth)

            print(out_dev_x.shape)
            for i in range(len(out_dev_x)):
                meter.add(out_dev_x[i], truth)
            print("auc middle", meter.value(0.05))
            print("TIME END", time.time() - start_time)
        print("AUC DONE", meter.value(0.05))

    meter = AUCMeter()
    # Switch between the two datasets
    for batch in tqdm(data_loader):
        count += 1
        x = autograd.Variable(batch["x"])
        y = autograd.Variable(torch.zeros(batch["x"].shape[0]))
        if args.cuda:
            x = x.cuda()
            y = y.cuda()

        pad_title = batch["pad_title"]
        pad_body = batch["pad_body"]

        if is_training:
            optimizer_feature.zero_grad()

        if args.model == "lstm":
            hidden = model.init_hidden(batch["x"].shape[1])
            hidden = repackage_hidden(hidden)
            out, embeddings = model(x, pad_title, pad_body, hidden)
        else:
            out, embeddings = model(x, pad_title, pad_body)

        # out, embeddings = model(x, pad_title, pad_body)
        
        loss_feature = F.multi_margin_loss(out, y.long(), margin=args.margin)
        # loss_domain = F.binary_cross_entropy(predictions, dataset_y)
        # loss_domain = F.cross_entropy(domain_out, dataset_y.long())

        if is_training:
            # total_loss = loss_feature - args.lambda_val * loss_domain
            total_loss = loss_feature
            total_loss.backward()
            optimizer_feature.step()

        # if count % 400 == 0: evaluate()

            #print "Test Loss"
        print("FEATURE LOSS", loss_feature.data[0])
        losses_feature += loss_feature.data
        # losses_domain += loss_domain.data
    print("DEV DATA")
    evaluate(dev_data)
    print("TEST DATA")
    evaluate(dev_data)



def train_model(train_data, dev_data, test_data, model, domain_model, args):
    optimizer_feature = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(0, args.epochs):
        print("epoch: " + str(epoch))
        loss = run_epoch(train_data, dev_data, test_data, model, domain_model,
                         optimizer_feature, None, args, True)
