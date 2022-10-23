import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
import tqdm
from torch.autograd import Variable

import math
import torch.nn.functional as F

torch.random.manual_seed(0)
np.random.seed(0)
# parser = argparse.ArgumentParser("Transformer-LSTM")
# parser.add_argument("-data_path", type=str, default="/home/zwc/transformer/stocks/shangzheng.csv", help="dataset path")
#
# args = parser.parse_args()
time_step = 40


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=64, num_layers=6, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class AttnDecoder(nn.Module):

    def __init__(self, code_hidden_size, hidden_size, time_step):
        super(AttnDecoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step

        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
        self.attn2 = nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=code_hidden_size, out_features=1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=1)
        self.tilde = nn.Linear(in_features=self.code_hidden_size + 1, out_features=1)
        self.fc1 = nn.Linear(in_features=code_hidden_size + hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, h, y_seq):
        h_ = h.transpose(0, 1)
        batch_size = h.size(0)
        d = self.init_variable(1, batch_size, self.hidden_size)
        s = self.init_variable(1, batch_size, self.hidden_size)
        h_0 = self.init_variable(1, batch_size, self.hidden_size)
        h_ = torch.cat((h_0, h_), dim=0)

        for t in range(self.T):
            x = torch.cat((d, h_[t, :, :].unsqueeze(0)), 2)
            h1 = self.attn1(x)
            _, states = self.lstm(y_seq[:, t].unsqueeze(0).unsqueeze(2), (h1, s))
            d = states[0]
            s = states[1]
        y_res = self.fc2(self.fc1(torch.cat((d.squeeze(0), h_[-1, :, :]), dim=1)))
        return y_res

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.permute(1, 0, 2)


# class StockDataset(Dataset):
#     def __init__(self, file_path, T=time_step, train_flag=True):
#         # read data
#         with open(file_path, "r", encoding="GB2312") as fp:
#             data_pd = pd.read_csv(fp)
#         self.train_flag = train_flag
#         self.data_train_ratio = 0.3
#         self.T = T  # use 10 data to pred
#         if train_flag:
#             self.data_len = int(self.data_train_ratio * len(data_pd))
#             data_all = np.array(data_pd['label'])
#             #data_all = (data_all - np.mean(data_all)) / np.std(data_all)
#             self.data = data_all[:self.data_len]
#         else:
#             self.data_len = int((1 - self.data_train_ratio) * len(data_pd))
#             data_all = np.array(data_pd['label'])
#             data_all = (data_all - np.mean(data_all)) / np.std(data_all)
#             self.data = data_all[-self.data_len:]
#         print("data len:{}".format(self.data_len))
#
#     def __len__(self):
#         return self.data_len - self.T
#
#     def __getitem__(self, idx):
#         return self.data[idx:idx + self.T], self.data[idx + self.T]
class StockDataset(Dataset):
    def __init__(self, file_path, T=time_step, train_flag=True):
        # read data
        with open(file_path, "r", encoding="GB2312") as fp:
            data_pd = pd.read_csv(fp)
        self.train_flag = train_flag
        self.data_train_ratio = 0.9
        self.T = T  # use 10 data to pred
        if train_flag:
            self.data_len = int(self.data_train_ratio * len(data_pd))
            data_all = np.array(data_pd['close'])
            data_all = (data_all - np.mean(data_all)) / np.std(data_all)
            self.data = data_all[:self.data_len]
        else:
            self.data_len = int((1 - self.data_train_ratio) * len(data_pd))
            data_all = np.array(data_pd['close'])
            data_all = (data_all - np.mean(data_all)) / np.std(data_all)
            self.data = data_all[-self.data_len:]
        print("data len:{}".format(self.data_len))

    def __len__(self):
        return self.data_len - self.T

    def __getitem__(self, idx):
        return self.data[idx:idx + self.T], self.data[idx + self.T]
