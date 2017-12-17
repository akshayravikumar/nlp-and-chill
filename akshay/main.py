import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import argparse

from eval_data import *
from train_data import *
from android_data import *
from shuffle_data import *

from run_model import *
from parse_utils import *
from cnn_model import *
from lstm_model import *
from domain_model import *

parser = argparse.ArgumentParser(description="NLP project P1")
parser.add_argument("--lr", type=float, default=1e-4) # 
parser.add_argument("--lr_domain", type=float, default=1e-4) #
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=60)
parser.add_argument("--margin", type=float, default=0.1) #
parser.add_argument("--weight", type=float, default=1e-3) #
parser.add_argument("--lambda_val", type=float, default=1e-2) #
parser.add_argument("--num_training_samples", type=int, default=NUM_SAMPLES)
parser.add_argument("--model", type=str, default="cnn") #

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

print(args)
print(args.lambda_val)

if args.model == "cnn":
    net = CNN(500, 3, args)
    domain_model = DomainClassifier(500, args)
elif args.model == "lstm":
    net = LSTM(300, 300)
    domain_model = DomainClassifier(300, args)
else:
    raise Exception("Model is either cnn or lstm")


if args.cuda:
    net = net.cuda()
    domain_model = domain_model.cuda()


print("Parsing Ubuntu data...")
train = make_set(SOURCE_DIR + "train_random.txt", args, training=True)
dev = make_set(SOURCE_DIR + "dev.txt", args)
test = make_set(SOURCE_DIR + "test.txt", args)

train_data = TrainData(train)
dev_data = EvalData(dev)
test_data = EvalData(test)

print("Parsing Android data...")


train_data_target = AndroidData(size=22)
dev_data_target = AndroidData(pos_map=dev_pos, neg_map=dev_neg)
test_data_target = AndroidData(pos_map=test_pos, neg_map=test_neg)

shuffle_data = ShuffleData(train_data, train_data_target)

print()
print("Running model...")
results = train_model(shuffle_data, dev_data_target, test_data_target, net, domain_model, args)

