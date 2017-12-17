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
        domain_model.train()
    else:
        model.eval()
        domain_model.eval()

    dev_batch_size = 10
    count = 0
    def evaluate(dataset):
        meter = AUCMeter()
        dev_loader = torch.utils.data.DataLoader(dataset, batch_size=dev_batch_size, shuffle=True)
        start_time = time.time()
        for dev_batch in tqdm(dev_loader):
            model.eval()
            domain_model.eval()

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

            out_dev_x = out_dev_x_raw.data
            truth = [0] * len(out_dev_x[0])
            truth[0] = 1
            truth = np.asarray(truth)

            for i in range(len(out_dev_x)):
                meter.add(out_dev_x[i], truth)
            print("auc middle", meter.value(0.05))
        print("AUC DONE", meter.value(0.05))
    # Switch between the two datasets
    for batch in tqdm(data_loader):
        count += 1
        x = autograd.Variable(batch["x"])
        y = autograd.Variable(torch.zeros(batch["x"].shape[0]))
        dataset_y = autograd.Variable(batch["dataset"]).float()
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
            dataset_y = dataset_y.cuda()

        pad_title = batch["pad_title"]
        pad_body = batch["pad_body"]

        if is_training:
            optimizer_domain.zero_grad()
            optimizer_feature.zero_grad()
         
        if args.model == "lstm":
            hidden = model.init_hidden(batch["x"].shape[1])
            hidden = repackage_hidden(hidden)
            out, embeddings = model(x, pad_title, pad_body, hidden)
        else:
            out, embeddings = model(x, pad_title, pad_body)
        
        print(embeddings[:,0,:].shape)
        # out, embeddings = model(x, pad_title, pad_body)
        domain_out = domain_model(embeddings[:,0,:])
        # preds = []
        # for i in range(len(domain_out.data)):
        #    if domain_out.data[i][0] >= 0.5:
        #        preds.append(0)
        #    else:
        #        preds.append(1)
        
        # predictions = autograd.Variable(torch.from_numpy(np.asarray(preds)).float())
        # if args.cuda:
        #     predictions = predictions.cuda()

        loss_feature = F.multi_margin_loss(out, y.long(), margin=args.margin)
        # loss_domain = F.binary_cross_entropy(predictions, dataset_y)
        loss_domain = F.cross_entropy(domain_out, dataset_y.long())

        if is_training:
            print("backprop")
            total_loss = loss_feature - args.lambda_val * loss_domain
            total_loss.backward()
            optimizer_feature.step()
            optimizer_domain.step()

        # if count % 400 == 0: evaluate()

            #print "Test Loss"
        print("FEATURE LOSS", loss_feature.data[0])
        print("DOMAIN LOSS", loss_domain.data[0])
        losses_feature += loss_feature.data
        losses_domain += loss_domain.data
    print("DEV DATA")
    evaluate(dev_data)
    print("TEST DATA")
    evaluate(test_data)



def train_model(train_data, dev_data, test_data, model, domain_model, args):
    optimizer_feature = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_domain = torch.optim.Adam(domain_model.parameters(), lr=-args.lr_domain)

    for epoch in range(0, args.epochs):
        print("epoch: " + str(epoch))
        loss = run_epoch(train_data, dev_data, test_data, model, domain_model,
                         optimizer_feature, optimizer_domain, args, True)
