import torch
import torch.nn as nn
import math


class RNN(nn.Module):
    def __init__(self, args, num_classes):
        super(RNN, self).__init__()
        self.device = torch.device(f'cuda:{args.gpu_num}')
        self.embedding_layer = nn.Linear(300, 512)
        self.num_layers = 3

        self.hidden_size = 512

        self.layer_norm0 = torch.nn.LayerNorm(self.hidden_size)
        self.layer_norm1 = torch.nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = torch.nn.LayerNorm(self.hidden_size)

        self.wx0 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.wh0 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.wx1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.wh1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.wx2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.wh2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)


        self.tanh = nn.Tanh()

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, num_classes)
        )
    def forward(self, x):
        x = self.embedding_layer(x)
        h_t_0 = torch.zeros(x.size()[0], self.hidden_size).to(self.device)
        h_t_1 = torch.zeros(x.size()[0], self.hidden_size).to(self.device)
        h_t_2 = torch.zeros(x.size()[0], self.hidden_size).to(self.device)
        for i in range(20):
            v_t = x[:,i,:]
            v_t = torch.squeeze(v_t, 1)
            h_t_0 = self.tanh(self.layer_norm0(self.wx0(v_t) + self.wh0(h_t_0)))
            h_t_1 = self.tanh(self.layer_norm1(self.wx1(h_t_0) + self.wh1(h_t_1)))
            h_t_2 = self.tanh(self.layer_norm2(self.wx2(h_t_1) + self.wh2(h_t_2)))

        logit = self.fc(h_t_2)
        return logit


class BidirectionalRNN(nn.Module):
    def __init__(self, args, num_classes):
        super(BidirectionalRNN, self).__init__()
        self.device = torch.device(f'cuda:{args.gpu_num}')
        self.embedding_layer = nn.Linear(300, 512)
        self.num_layers = 3
        self.hidden_size = 512

        self.layer_norm0 = torch.nn.LayerNorm(self.hidden_size)
        self.layer_norm1 = torch.nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = torch.nn.LayerNorm(self.hidden_size)

        self.drop_out = nn.Dropout2d(p=0.2)


        self.wx0_l2r = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.wh0_l2r = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.wx1_l2r = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.wh1_l2r = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.wx2_l2r = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.wh2_l2r = nn.Linear(self.hidden_size, self.hidden_size, bias=True)


        self.wx0_r2l = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.wh0_r2l = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.wx1_r2l = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.wh1_r2l = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.wx2_r2l = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.wh2_r2l = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.tanh = nn.Tanh()
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, num_classes)
        )

    def forward(self, x, reverse_x, pad_start_index, max_length):
        x_l2r = self.embedding_layer(x)
        x_r2l = self.embedding_layer(reverse_x)
        h_t_0_l2r = torch.zeros(x.size()[0], self.hidden_size).to(self.device)
        h_t_1_l2r = torch.zeros(x.size()[0], self.hidden_size).to(self.device)
        h_t_2_l2r = torch.zeros(x.size()[0], self.hidden_size).to(self.device)

        h_t_0_r2l = torch.zeros(x.size()[0], self.hidden_size).to(self.device)
        h_t_1_r2l = torch.zeros(x.size()[0], self.hidden_size).to(self.device)
        h_t_2_r2l = torch.zeros(x.size()[0], self.hidden_size).to(self.device)

        out_l2r = torch.tensor([])
        out_r2l = torch.tensor([])

        for i in range(max_length):
            v_t_l2r = x_l2r[:,i,:]
            v_t_l2r = torch.squeeze(v_t_l2r, 1)
            h_t_0_l2r = self.drop_out(self.tanh(self.layer_norm0(self.wx0_l2r(v_t_l2r) + self.wh0_l2r(h_t_0_l2r))))
            h_t_1_l2r = self.drop_out(self.tanh(self.layer_norm1(self.wx1_l2r(h_t_0_l2r) + self.wh1_l2r(h_t_1_l2r))))
            h_t_2_l2r = self.drop_out(self.tanh(self.layer_norm2(self.wx2_l2r(h_t_1_l2r) + self.wh2_l2r(h_t_2_l2r))))
            unsqueeze_h_t_2_l2r = torch.unsqueeze(h_t_2_l2r, 1)
            if len(out_l2r) == 0:
                out_l2r = unsqueeze_h_t_2_l2r
            else:
                out_l2r = torch.cat([out_l2r, unsqueeze_h_t_2_l2r], dim=1)

        for i in range(max_length):
            v_t_r2l = x_r2l[:, i, :]
            v_t_r2l = torch.squeeze(v_t_r2l, 1)
            h_t_0_r2l = self.drop_out(self.tanh(
                self.layer_norm0(self.wx0_r2l(v_t_r2l) + self.wh0_r2l(h_t_0_r2l))))
            h_t_1_r2l = self.drop_out(self.tanh(
                self.layer_norm1(self.wx1_r2l(h_t_0_r2l) + self.wh1_r2l(h_t_1_r2l))))
            h_t_2_r2l = self.drop_out(self.tanh(
                self.layer_norm2(self.wx2_r2l(h_t_1_r2l) + self.wh2_r2l(h_t_2_r2l))))
            unsqueeze_h_t_2_r2l = torch.unsqueeze(h_t_2_r2l, 1)
            if len(out_r2l) == 0:
                out_r2l = unsqueeze_h_t_2_r2l
            else:
                out_r2l = torch.cat([out_r2l, unsqueeze_h_t_2_r2l], dim=1)
        concat_vector = self.ordered_concat(out_l2r, out_r2l, pad_start_index)
        logit = self.fc(concat_vector)
        logit = logit.view(-1, logit.size()[-1])
        return logit


    def ordered_concat(self, l2r, r2l, pad_start_index):
        concat_vectors = []
        for i in range(len(l2r)):
            concat_vector = []
            p = pad_start_index[i]
            for j in range(0, p):
                v = torch.cat((l2r[i][j], r2l[i][p-j-1]), dim=-1)
                concat_vector.append(v)
            for j in range(p, len(l2r[0])):
                v = torch.cat((l2r[i][j], r2l[i][j]), dim=-1)
                concat_vector.append(v)
            concat_vector = torch.stack(concat_vector, dim=0)
            concat_vectors.append(concat_vector)
        concat_vectors = torch.stack(concat_vectors, dim=0)

        return concat_vectors