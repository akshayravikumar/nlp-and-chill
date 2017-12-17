import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import argparse

from load_data import *
from run_model2 import *
from parse_utils import *
from cnn_model import *
from lstm_model import *

parser = argparse.ArgumentParser(description = "NLP project P1")
parser.add_argument("--lr", type = float, default = 0.001)
parser.add_argument("--epochs", type = int, default = 10)
parser.add_argument("--batch_size", type = int, default = 40)
parser.add_argument("--weight", type = float, default = 1e-3)
parser.add_argument("--num_training_samples", type = int, default = NUM_SAMPLES)
parser.add_argument("--model", type = str, default = "cnn")

args = parser.parse_args()

# print("Parsing data...")
# if args.model == "cnn":
# 	net = CNN(500, 3)
# elif args.model == "lstm":
# 	net = LSTM(500)
# else:
# 	raise Exception("Model is either cnn or lstm")
net=LSTM(300,200).cuda()

train, dev, test = make_sets(
    DATA_DIR + "train_random.txt",
    DATA_DIR + "dev.txt",
    DATA_DIR + "test.txt",
    args
)

train_data = Data(train)
dev_data = Eval_Data(dev)
test_data = Eval_Data(test)


print()
print("Running model...")
results = train_model(train_data, dev_data, test_data, net, args)
