import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np

def run_epoch(data, model, optimizer, args, is_training):
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
        print("LOSS", loss.data[0])
        losses += loss.data
    return 1.0 * losses/count

def train_model(train_data, dev_data, test_data, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    for epoch in range(0, args.epochs):
        print("epoch: " + str(epoch))
        loss = run_epoch(train_data, model, optimizer, args, True)
