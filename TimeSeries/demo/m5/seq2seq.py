# -*- coding: utf-8 -*-
# @Time : 2022/4/27 15:20
# @Author : li.zhang
# @File : seq2seq.py

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from fastprogress import master_bar, progress_bar
import joblib
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, embedding_dim
        self.num_layers = 3
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.35
        )

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(device))

        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(device))

        # x(batch_size, seq_len, hidden_size)
        # hidden(number_layer, batch_size, hidden_size)
        # cell(number_layer, batch_size, hidden_size)
        x, (hidden, cell) = self.rnn1(x, (h_1, c_1))

        # return hidden_n.reshape((self.n_features, self.embedding_dim))
        return x, hidden, cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        [batch_size, seq_len, (hidden_output)]:[1, 90, 1024] ---linear---> [1, 90, 512] ---linear---> [1, 90, 1]
        :param hidden:
        :param encoder_outputs:
        :return:
        """
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden[2:3, :, :]  # The last layers [1:1:512] [layers, batch_size, hidden_size]

        # print("hidden size is",hidden.size())

        # repeat decoder hidden state src_len times
        # hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        hidden = hidden.repeat(1, src_len, 1)  # copy step_time times [1, 90, 512]
        # example: [[[-1, -2, -3, -4], [-1, -2, -3, -4], [-1, -2, -3, -4], [-1, -2, -3, -4], ...]]  复制90次[1, 90, 512]

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # print("encode_outputs size after permute is:",encoder_outputs.size())

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        # cat [[[-1, -2, -3, -4, e_1, e_2, e_3, e_4, ...], [-1, -2, -3, -4, e_1, e_2, e_3, e_4, ...]]]  [1, 90, 1024]-> linear [1, 90, 512]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # concat dim

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)  # [1, 90, 1] -> [1, 90]

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=1,
            hidden_size=input_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.35
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x, input_hidden, input_cell):
        x = x.reshape((1, 1, 1))

        x, (hidden_n, cell_n) = self.rnn1(x, (input_hidden, input_cell))

        x = self.output_layer(x)
        return x, hidden_n, cell_n


class AttentionDecoder(nn.Module):
    def __init__(self, seq_len, attention, input_dim=64, n_features=1, encoder_hidden_state=512):
        super(AttentionDecoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = input_dim, n_features
        self.attention = attention

        self.rnn1 = nn.LSTM(
            # input_size=1,
            input_size=encoder_hidden_state + 1,  # Encoder Hidden State + One Previous input
            hidden_size=input_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.35
        )

        self.output_layer = nn.Linear(self.hidden_dim * 2, n_features)

    def forward(self, x, input_hidden, input_cell, encoder_outputs):
        a = self.attention(input_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs) # [1, 1, 90]*[1, 90, 512]  [1, 1, 512]

        x = x.reshape((1, 1, 1))

        rnn_input = torch.cat((x, weighted), dim=2)  # [1, 1, 513]

        # x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))
        x, (hidden_n, cell_n) = self.rnn1(rnn_input, (input_hidden, input_cell))

        output = x.squeeze(0)
        weighted = weighted.squeeze(0)

        x = self.output_layer(torch.cat((output, weighted), dim=1))
        return x, hidden_n, cell_n


class Seq2Seq(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64, output_length=28):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.attention = Attention(512, 512)
        self.output_length = output_length
        self.decoder = AttentionDecoder(seq_len, self.attention, embedding_dim, n_features).to(device)

    def forward(self, x, prev_y):
        encoder_output, hidden, cell = self.encoder(x)

        # Prepare place holder for decoder output
        targets_ta = []
        # prev_output become the next input to the LSTM cell
        prev_output = prev_y

        # itearate over LSTM - according to the required output days
        for out_days in range(self.output_length):
            prev_x, prev_hidden, prev_cell = self.decoder(prev_output, hidden, cell, encoder_output)
            hidden, cell = prev_hidden, prev_cell
            prev_output = prev_x

            targets_ta.append(prev_x.reshape(1))

        targets = torch.stack(targets_ta)

        return targets



def train_model(model, TrainX, Trainy, ValidX, Validy, seq_length, n_epochs):
    history = dict(train=[], val=[])

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    mb = master_bar(range(1, n_epochs + 1))

    for epoch in mb:
        model = model.train()

        train_losses = []
        for i in progress_bar(range(TrainX.size()[0]), parent=mb):
            seq_inp = TrainX[i, :, :].to(device)
            seq_true = Trainy[i, :, :].to(device)

            optimizer.zero_grad()

            seq_pred = model(seq_inp, seq_inp[seq_length - 1:seq_length, :])

            loss = criterion(seq_pred, seq_true)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for i in progress_bar(range(validX.size()[0]), parent=mb):
                seq_inp = ValidX[i, :, :].to(device)
                seq_true = Validy[i, :, :].to(device)

                seq_pred = model(seq_inp, seq_inp[seq_length - 1:seq_length, :])

                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("saved best model epoch:", epoch, "val loss is:", val_loss)

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        scheduler.step()
    # model.load_state_dict(best_model_wts)
    return model.eval(), history

INPUT = 'C:/ZhangLI/Codes/DataSet/m5-forecasting-accuracy'
# INPUT = 'E:/Dataset/m5-forecasting-accuracy/'
trainX = joblib.load(f'{INPUT}/trainX.pkl')
trainy = joblib.load(f'{INPUT}/trainy.pkl')
validX = joblib.load(f'{INPUT}/validX.pkl')
validy = joblib.load(f'{INPUT}/validy.pkl')

n_features = 1
seq_length = 90
labels_length =28  # 使用90天预测28天

model = Seq2Seq(seq_length, n_features, 512)

print(model)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=4e-3, weight_decay=1e-5)
criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5e-3, eta_min=1e-8, last_epoch=-1)

model, history = train_model(
    model,
    trainX, trainy,
    validX, validy,
    seq_length,
    n_epochs=30,  ## Training only on 30 epochs to save GPU time

)