# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from time import time
import gc
import warnings
warnings.filterwarnings("ignore")
import logging

logging.basicConfig(
    filename="narm.log",
    level=logging.INFO,
    format='[%(asctime)s] - %(message)s'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


df_sess = pd.read_csv('data/sessions_train.csv')
df_test = pd.read_csv('data/sessions_test_task1.csv')
print(f'df_sess.shape = {df_sess.shape}, df_test.shape = {df_test.shape}')
logging.info(f'df_sess.shape = {df_sess.shape}, df_test.shape = {df_test.shape}')
products = pd.read_csv('data/products_train.csv')
print(f'products.shape = {products.shape}')
logging.info(f'products.shape = {products.shape}')
product2idx = dict(zip(products['id'].unique(), range(1, products['id'].nunique()+1)))
idx2product = dict(zip(range(1, products['id'].nunique()+1), products['id'].unique()))
product_num = products['id'].nunique() + 1

product_dict = dict()
locales = ['UK', 'DE', 'JP', 'IT', 'FR', 'ES']

for locale in locales:
    product_dict[locale] = [product2idx[x] for x in list(products[products['locale']==locale]['id'].unique())]
    
def str2list(x):
    x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
    l = [product2idx[i] for i in x.split() if i]
    return l


df_sess['prev_items'] = df_sess['prev_items'].apply(lambda x: str2list(x))
df_test['prev_items'] = df_test['prev_items'].apply(lambda x: str2list(x))

df_sess['next_item'] = df_sess['next_item'].apply(lambda x: product2idx[x])

df_train, df_valid, _, _ = train_test_split(
    df_sess, df_sess['locale'], test_size=0.1, random_state=2023, stratify=df_sess['locale'])

print(f'df_train.shape = {df_train.shape}, df_valid.shape = {df_valid.shape}')
logging.info(f'df_train.shape = {df_train.shape}, df_valid.shape = {df_valid.shape}')
train = (list(df_train["prev_items"]), list(df_train["next_item"]))
valid = (list(df_valid["prev_items"]), list(df_valid["next_item"]))


def collate_fn_train(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(hist) for hist, _ in data]
    labels = []
    padded_seq = torch.zeros(len(data), max(lens)).long()
    for i, (hist, label) in enumerate(data):
        padded_seq[i, :lens[i]] = torch.LongTensor(hist)
        labels.append(label)

    return padded_seq, torch.tensor(labels).long(), lens

class TrainDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return len(self.data[0])

def TrainDataLoader(data, bs=512):
    data_set = TrainDataset(data)
    data_loader = DataLoader(data_set, batch_size=bs, shuffle=True, collate_fn=collate_fn_train, drop_last=True)

    return data_loader


class Model(nn.Module):

    def __init__(self, n_items, hidden_size, embedding_dim, n_layers=1):
        super(Model, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(self.n_items + 1, self.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.25) # 0.25
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers, batch_first=True)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5) # 0.5
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        # self.sf = nn.Softmax()
        self.device = device

        # self.tanh = nn.Tanh()

    def forward(self, seq, lengths):
        hidden = self.init_hidden(seq.size(0))
        embs = self.emb_dropout(self.emb(seq))
        embs = pack_padded_sequence(embs, lengths, batch_first=True)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out, batch_first=True)

        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        # gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())
        q2 = self.a_2(ht)
        mask = torch.where(seq > 0, torch.tensor([1.], device=self.device),
                           torch.tensor([0.], device=self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)

        # c_t = self.tanh(c_t)

        item_embs = self.emb(torch.arange(self.n_items + 1).to(self.device))
        scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0))
        # scores = self.sf(scores)

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

class Loss(nn.Module):
    def __init__(self, reg=0, eps=1e-6):
        super(Loss, self).__init__()
        self.reg = reg
        self.eps = eps

    def forward(self, p, n):
        p = torch.exp(p)
        n = torch.exp(n)
        prob = - torch.log(p / (p + torch.sum(n, dim=1, keepdim=True)) + self.eps)

        return prob.sum() + self.reg


def evaluate(rec_matrix, targets, match_num):
    target_repeats = torch.repeat_interleave(targets.view(-1, 1), dim=1, repeats=match_num)
    judge = torch.where(rec_matrix - target_repeats == 0)
    hit = len(judge[0])
    mrr = 0
    ndcg = 0
    for pos in judge[1]:
        mrr += 1 / (pos.float() + 1)
        ndcg += 1 / torch.log2(pos.float() + 2)

    return hit, ndcg, mrr


item_nuniq = product_num
emb_dim = 64
epochs = 10
lr = 1e-3
hidden_size = 100
n_layers = 1
match_num = 100
gamma = 1e-5
mix_recall_num = 100
bs = 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(item_nuniq, hidden_size, emb_dim, n_layers).to(device)

num_params = sum(param.numel() for param in model.parameters())
reg = gamma * num_params
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Start training:")

train_loader = TrainDataLoader(train, bs)
valid_loader = TrainDataLoader(valid, bs=512)

best_mrr = 0
for epoch in range(epochs):
    if epoch == 0:
        st = time()
    print('lr:%.4e' % optimizer.param_groups[0]['lr'])
    logging.info('lr:%.4e' % optimizer.param_groups[0]['lr'])
    model.train()
    for i, (hist_click, target, lens) in enumerate(train_loader):
        hist_click, target = hist_click.to(device), target.to(device)
        output = model(hist_click, lens)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('Epoch: [{0}][{1}/{2}]\t Loss {loss:.4f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))
        logging.info('Epoch: [{0}][{1}/{2}]\t Loss {loss:.4f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))
    model.eval()
    
    torch.save(model, f'narm_{epoch}.pkl')
    HIT, NDCG, MRR = 0, 0, 0
    length = 0
    for hist_click, target, lens in valid_loader:
        hist_click, target = hist_click.to(device), target.to(device)
        candidates_score = F.softmax(model(hist_click, lens)[:, 1:], dim=1)
        candidate_argsort = candidates_score.argsort(dim=1, descending=True)
        rec_matrix = candidate_argsort[:, :match_num] + 1
        hit, ndcg, mrr = evaluate(rec_matrix, target, match_num)
        length += len(rec_matrix)
        HIT += hit
        NDCG += ndcg
        MRR += mrr
    HIT /= length
    NDCG /= length
    MRR /= length
    print('[+] HIT@{} : {}'.format(match_num, HIT))
    print('[+] NDCG@{} : {}'.format(match_num, NDCG))
    print('[+] MRR@{} : {}'.format(match_num, MRR))
    logging.info('[+] HIT@{} : {}'.format(match_num, HIT))
    logging.info('[+] NDCG@{} : {}'.format(match_num, NDCG))
    logging.info('[+] MRR@{} : {}'.format(match_num, MRR))
    if best_mrr < MRR:
        best_model = model
        best_mrr = MRR
        corr_hit, corr_ndcg = HIT, NDCG
        torch.save(best_model, 'narm.pkl')
        print('Model updated.')
    if epoch == 0:
        et = time()
        print('The last time of 1 epoch: {} min'.format((et - st) / 60))
