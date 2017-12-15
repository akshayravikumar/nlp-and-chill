import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from metrics import Measure
from meter import AUCMeter

LAMBDA = 0.1

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
        domain_model.train()

    count = 0

    meter = AUCMeter()
    # Switch between the two datasets
    for batch in tqdm(data_loader):
        count += 1
        x = autograd.Variable(batch["x"])
        y = autograd.Variable(torch.zeros(batch["x"].shape[0]))
        dataset_y = autograd.Variable(batch["dataset"]).float()

        pad_title = batch["pad_title"]
        pad_body = batch["pad_body"]

        if is_training:
            optimizer_domain.zero_grad()
            optimizer_feature.zero_grad()

        out, embeddings = model(x, pad_title, pad_body)
        # print("out data shape", out.data.shape)
        # print("embeddings", embeddings.data.shape)
        domain_out = domain_model(embeddings[:,0,:])
        # print("domain out", domain_out.data)
        preds = []
        for i in range(len(domain_out.data)):
            if domain_out.data[i][0] >= 0.5:
                preds.append(0)
            else:
                preds.append(1)
        
        predictions = autograd.Variable(torch.from_numpy(np.asarray(preds)).float())
        # print("predictions", predictions)
        # print("dataset_y", dataset_y)

        loss_feature = F.multi_margin_loss(out, y.long(), margin=0.2)
        loss_domain = F.binary_cross_entropy(predictions, dataset_y)

        if is_training:
            total_loss = loss_feature - LAMBDA * loss_domain
            total_loss.backward()
            optimizer_feature.step()
            optimizer_domain.step()

        if count % 5 == 1:
            dev_loader = torch.utils.data.DataLoader(
                dev_data, batch_size=1, shuffle=True)
            #test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False)
            batch_row = 0
            for dev_batch in tqdm(dev_loader):
                dev_x = autograd.Variable(dev_batch["x"])
                dev_pad_title = dev_batch["pad_title"]
                dev_pad_body = dev_batch["pad_body"]
                out_dev_x_raw, _ = model(dev_x, dev_pad_title, dev_pad_body)
                out_dev_x = out_dev_x_raw.data[0]
                truth = [0] * len(out_dev_x)
                truth[0] = 1
                # print("OUT DEV X", out_dev_x)
                # print(out_dev_x.shape, len(truth))

                meter.add(out_dev_x, np.asarray(truth))
                #out_dev_x = dev_batch["BM25"][0,:]
                # print(out_dev_x, ids[1:], pos_ids)
                batch_row += 1
            #print "Test Loss"
        print("AUC", meter.value(0.05))
        print("FEATURE LOSS", loss_feature.data[0])
        print("DOMAIN LOSS", loss_domain.data[0])
        losses_feature += loss_feature.data
        losses_domain += loss_domain.data
    return 1.0 * losses / count


def train_model(train_data, dev_data, test_data, model, domain_model, args):
    optimizer_feature = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_domain = torch.optim.Adam(
        domain_model.parameters(), lr=-args.lr_domain)

    for epoch in range(0, args.epochs):
        print("epoch: " + str(epoch))
        loss = run_epoch(train_data, dev_data, test_data, model, domain_model,
                         optimizer_feature, optimizer_domain, args, True)
