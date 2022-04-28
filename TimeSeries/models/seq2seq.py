# -*- coding: utf-8 -*-
# @Time : 2022/4/28 14:48
# @Author : li.zhang
# @File : seq2seq.py


from utils.pyp import *



class Encoder(nn.Module):
    def __init__(self, args, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.args = args
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
        # x = x.reshape((1, self.seq_len, self.n_features))

        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(self.args['device']))

        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).to(self.args['device']))

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

        hidden = hidden[-1:, :, :]  # The last layers (1,32,64) (layers, batch_size, hidden_size)

        # print("hidden size is",hidden.size())

        # repeat decoder hidden state src_len times
        # hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        hidden = hidden.repeat(src_len, 1, 1)  # copy step_time times (1,32,64)  output(32, 300, 64)
        hidden = hidden.permute(1, 0, 2)
        # example: [[[-1, -2, -3, -4], [-1, -2, -3, -4], [-1, -2, -3, -4], [-1, -2, -3, -4], ...]]  复制90次[1, 90, 512]

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # print("encode_outputs size after permute is:",encoder_outputs.size())

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        # cat [[[-1, -2, -3, -4, e_1, e_2, e_3, e_4, ...], [-1, -2, -3, -4, e_1, e_2, e_3, e_4, ...]]]  [1, 90, 1024]-> linear [1, 90, 512]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # concat dim

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy)  # [1, 90, 1] -> [1, 90]
        attention = attention.squeeze(2)
        attention = torch.add(attention, 1)

        # attention= [batch size, src len]
        a = F.softmax(attention, dim=1)

        return a



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

        weighted = torch.bmm(a, encoder_outputs) # [32, 1, 300]*[32, 300, 512]  [32, 1, 512]

        # x = x.reshape((1, 1, 1))

        rnn_input = torch.cat((x, weighted), dim=2)  # [32, 1, 513]

        # x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))
        x, (hidden_n, cell_n) = self.rnn1(rnn_input, (input_hidden, input_cell))

        output = x.squeeze(1)
        weighted = weighted.squeeze(1)

        x = self.output_layer(torch.cat((output, weighted), dim=1))
        return x, hidden_n, cell_n


class Seq2Seq(nn.Module):
    """
    Desc: build seq2seq model
    """
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.args = args
        seq_len = self.args['seq_len']  # 300
        n_features = self.args['n_features']  # 10
        embedding_dim = self.args['embedding_dim']  # 64
        output_length = self.args['output_length']  # 288
        device = self.args['device']

        self.encoder = Encoder(self.args, seq_len, n_features, embedding_dim).to(device)
        self.attention = Attention(512, 512)
        self.output_length = output_length
        self.decoder = AttentionDecoder(seq_len, self.attention, embedding_dim, n_features).to(device)


    def forward(self, x, prev_y):
        encoder_output, hidden, cell = self.encoder(x)  # x (32, 300, 10)

        # Prepare place holder for decoder output
        targets_ta = []
        # prev_output become the next input to the LSTM cell
        prev_output = prev_y

        # itearate over LSTM - according to the required output days
        for out_days in range(self.output_length):
            prev_x, prev_hidden, prev_cell = self.decoder(prev_output, hidden, cell, encoder_output)
            hidden, cell = prev_hidden, prev_cell
            prev_output = prev_x.unsqueeze(1)[:, :, -1:]

            targets_ta.append(prev_x[:, -1:])

        targets = torch.stack(targets_ta)

        return targets