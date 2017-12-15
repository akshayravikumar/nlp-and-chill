import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import numpy as np

SOURCE_DIR = "askubuntu/"
TARGET_DIR = "Android/"

NUM_SAMPLES = 12724


# Returns an array of vecs for words and their mappings (word to index in array)
def embeddings(directory):
    mapping = {}
    file = open(directory)
    embeddings = []
    count = 0
    elt = file.readline()
    while elt != "":
        if count % 10000 == 0:
            print(count)
        arr = elt.split()
        word, vector = arr[0], arr[1:]
        mapping[word] = count
        temp = [float(x) for x in vector]
        embeddings.append(temp)

        elt = file.readline()
        count += 1
    return embeddings, mapping


def parse_file(file, a, b, c):
    matrix = []

    query = file.readline()
    while query != "":
        temp = np.zeros((b, c))
        if "\t\t" in query:
            query = file.readline()
            continue
        arr = query.split("\t")

        # An extra field is found in val and test sets (BM25 scores)
        if len(arr) == 4:
            q, q_pos, Q_neg, BM = arr
            BM = BM.split()
            BM = [float(x) for x in BM]
            temp[3, :] = BM[:20]
        else:
            q, q_pos, Q_neg = arr
        q = int(q)
        q_pos = q_pos.split()
        q_pos = [int(x) for x in q_pos]
        Q_neg = Q_neg.split()
        Q_neg = [int(x) for x in Q_neg]

        q_pad = np.zeros((20, ), dtype="int") - 1
        q_pad[0] = q

        q_pos_pad = np.zeros((20, ), dtype="int") - 1
        q_pos_pad[:min(len(q_pos), 20)] = q_pos[:min(20, len(q_pos))]

        q_neg_pad = Q_neg[:20]

        temp[0, :] = q_pad
        temp[1, :] = q_pos_pad
        temp[2, :] = q_neg_pad

        query = file.readline()
        matrix.append(temp)

    return np.asarray(matrix)


# Assuming num_negs>num_pos
num_negs = 20


def make_set(file_name, args, training=False):
    file = open(file_name)
    if training:
        return parse_file(file, args.num_training_samples, 3, num_negs)
    return parse_file(file, 200, 4, num_negs)


# Takes raw corpus files and outputs tuple of dictionaries, maps from id to array of words(string)
def corpus(directory):
    id_to_title = {}
    id_to_body = {}
    file = open(directory)
    temp = file.readline()

    while temp != "":
        temp = temp.split("\t")
        id, title, body = temp
        id = int(id)

        title = title.split()
        body = body.split()

        id_to_title[id] = title
        id_to_body[id] = body

        temp = file.readline()
    return id_to_title, id_to_body

def parse_android(pos_filename, neg_filename):
    pos = open(pos_filename)
    neg = open(neg_filename)

    negative = {}
    positive = {}
    for line in pos:
        id1, id2 = [int(x) for x in line.split(" ")]
        if id1 not in positive:
            positive[id1] = []
        if len(positive[id1]) >= 1:
            continue
        positive[id1].append(id2)

    for line in neg:
        id1, id2 = [int(x) for x in line.split(" ")]
        if id1 not in negative:
            negative[id1] = []
        if len(negative[id1]) >= 100:
            continue
        negative[id1].append(id2)

    return positive, negative


dev_pos, dev_neg = parse_android(TARGET_DIR + "dev.pos.txt", TARGET_DIR + "dev.neg.txt")
test_pos, test_neg = parse_android(TARGET_DIR + "test.pos.txt", TARGET_DIR + "test.neg.txt")
print(max(len(v) for v in dev_pos.values()))
print(max(len(v) for v in test_pos.values()))

print("Parsing embeddings...")
embeddings, map = embeddings(SOURCE_DIR + "vectors_pruned.200.txt")

print("Parsing Ubuntu data...")
id_to_title, id_to_body = corpus(SOURCE_DIR + "text_tokenized.txt")

print("Parsing Android data...")
id_to_title_target, id_to_body_target = corpus(TARGET_DIR + "corpus.tsv")

# Question id, mapping: either id_to_title or id_to_body
def question_to_vec(question, mapping):
    sentence = mapping.get(question)
    matrix = []
    for word in sentence:
        if len(matrix) > 100 - 1:
            break
        if word in map:
            id = map.get(word)
            vec = embeddings[id]
            matrix.append(vec)

    a = np.asarray(matrix).T

    if len(a.shape) == 1:
        return None
    else:
        x, y = a.shape
        a = a.reshape(1, x, y)
    garbage = torch.from_numpy(a).float()
    return garbage

# Fit the small into the big. (Assuming the shape is 3 tuple), assuming padding happens
# in the 3rd access. TODO: It might be helpful to do this for other axis. Not needed ATM
window = 3

def pad(big, small):
    big_a, big_b, big_c = big.shape
    small_a, small_b, small_c = small.shape

    compromise = min(big_c, small_c)
    big[:, :, :compromise] = small[:, :, :compromise]

    if compromise > big_c - window:
        return big, [big_c - window]
    else:
        return big, [compromise]
