import pandas as pd
import torch.nn as nn
from nltk.tokenize import word_tokenize
import json
import torch
import model
import argparse
import torch.optim as optim
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--gpu-num', default=3, type=int)
parser.add_argument('--task', default='classification', type=str, choices=['classification', 'pos'])
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--do-train', action='store_true', default=False)
args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu_num}')

criterion = nn.CrossEntropyLoss()
isDebug = True


def tokenization(sents):
    tokens = []
    for sent in sents:
        token = word_tokenize(sent)
        tokens.append(token)
    return tokens


def pad_sequence(tokens, max_length, pre_padding=True):
    new_tokens = []
    pad_start_index = []
    for token in tokens:
        # pre-sequence truncation
        if len(token) > max_length:
            new_token = token[:max_length]
            pad_start_index.append(max_length)

        elif len(token) < max_length:
            addition = max_length - len(token)
            # pre-padding
            if pre_padding:
                new_token = ['[PAD]'] * addition + token
                pad_start_index.append(0)
            # post-padding
            else:
                new_token = token + ['[PAD]'] * addition
                pad_start_index.append(len(token))
        else:
            new_token = token
            pad_start_index.append(len(token))
        new_tokens.append(new_token)
    return new_tokens, pad_start_index


def word2vec(tokens, word_dict, word_embedding_matrix):
    word_embs = []
    for token in tokens:
        word_emb = []
        for t in token:
            if t not in word_dict.keys():
                word_emb.append(word_embedding_matrix[word_dict['[UNK]']])
            else:
                word_emb.append(word_embedding_matrix[word_dict[t]])
        word_emb = torch.stack(word_emb, dim=0)
        word_embs.append(word_emb)
    word_embs = torch.stack(word_embs, dim=0)
    return word_embs


def flip_input(tokens, max_length):
    new_tokens = []
    for token in tokens:
        new_token = []
        if len(token) > max_length:
            token = token[:max_length]
        for t_idx in range(len(token)-1, -1, -1):
            new_token.append(token[t_idx])
        new_tokens.append(new_token)
    return new_tokens


def output2csv(pred_y, file_name='classification_class.pred.csv'):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for i, p in enumerate(pred_y):
            y_id = str(i)
            # if len(y_id) < 3:
            #     y_id = '0' * (3 - len(y_id)) + y_id
            writer.writerow(['S' + y_id, p[0].item()])
    print('file saved.')


# train classification
def classification_train(train_x, train_y, epoch):
    tr_loss = 0.
    correct = 0
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    batch_size = 256
    iteration = (len(train_x) // batch_size) + 1
    cur_i = 0
    for i in range(1, iteration + 1):
        if cur_i >= len(train_x):
            break
        if i < iteration:
            data, targets = train_x[cur_i:cur_i + batch_size].to(device), train_y[cur_i:cur_i + batch_size].to(device)
            cur_i += batch_size
        if i == iteration:
            data, targets = train_x[cur_i:].to(device), train_y[cur_i:].to(device)
            cur_i = len(train_x)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, targets)
        tr_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()

        print("\r[epoch {:3d}/{:3d}] loss: {:.6f}".format(epoch, args.epochs, loss), end=' ')

    tr_loss /= iteration
    tr_acc = correct / len(train_x)
    return tr_loss, tr_acc


# test classification
def classification_test(test_x, targets=None):
    net.eval()
    correct = 0.
    ts_loss = 0.
    with torch.no_grad():
        data = test_x.to(device)
        output = net(data)

        pred = output.argmax(dim=1, keepdim=True)
        if targets != None:
            targets = targets.to(device)
            ts_loss = criterion(output, targets)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            ts_acc = correct / len(test_x)
            return ts_loss, ts_acc
        else:
            return pred


# train pos-tagging
def pos_train(train_x, reverse_train_x, pad_start_index, max_length, train_y, epoch):
    tr_loss = 0.
    correct = 0
    all_samples = 0
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    batch_size = 256
    iteration = (len(train_x) // batch_size) + 1
    cur_i = 0
    for i in range(1, iteration + 1):
        if cur_i >= len(train_x):
            break
        if i < iteration:
            data, reverse_data, targets, pad_indexes = train_x[cur_i:cur_i + batch_size].to(device), reverse_train_x[cur_i:cur_i + batch_size].to(device),\
                                          train_y[cur_i:cur_i + batch_size].to(device), pad_start_index[cur_i:cur_i + batch_size]
            cur_i += batch_size
        if i == iteration:
            data, reverse_data, targets, pad_indexes = train_x[cur_i:].to(device), reverse_train_x[cur_i:].to(device),\
                                                       train_y[cur_i:].to(device), pad_start_index[cur_i:]
            cur_i = len(train_x)
        optimizer.zero_grad()
        output = net(data, reverse_data, pad_indexes, max_length)
        targets = targets.view(-1)
        all_samples += len(targets)
        # set mask
        mask = torch.where(targets == 0, False, True).unsqueeze(1).expand(-1, 18)
        output *= mask
        loss = criterion(output, targets)
        tr_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(targets.view_as(pred)).sum().item()

        print("\r[epoch {:3d}/{:3d}] loss: {:.6f}".format(epoch, args.epochs, loss), end=' ')

    tr_loss /= iteration
    tr_acc = correct / all_samples
    return tr_loss, tr_acc


# test pos-tagging
def pos_test(test_x, reverse_test_x, pad_start_index, max_length, targets=None):
    net.eval()
    correct = 0.
    ts_loss = 0.

    with torch.no_grad():
        if targets != None:
            data = test_x.to(device)
            reverse_data = reverse_test_x.to(device)
            output = net(data, reverse_data, pad_start_index, max_length)
            pred = output.argmax(dim=1, keepdim=True)
            targets = targets.to(device)
            targets = targets.view(-1)
            ts_loss = criterion(output, targets)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            ts_acc = correct / len(targets)
            return ts_loss, ts_acc
        else:
            outputs = None
            for i in range(len(test_x)):
                data = test_x[i].to(device)
                data = data.unsqueeze(0)
                reverse_data = reverse_test_x[i].to(device)
                reverse_data = reverse_data.unsqueeze(0)
                output = net(data, reverse_data, [pad_start_index[i]], max_length)
                output = output[:pad_start_index[i]]
                if outputs == None:
                    outputs = output
                else:
                    outputs = torch.cat((outputs, output), dim=0)
            pred = outputs.argmax(dim=1, keepdim=True)
            return pred




if __name__ == '__main__':
    print(args.task)

    # classification
    if args.task == 'classification':
        # load _data
        train_data = pd.read_csv('./dataset/classification/train_set.csv')
        test_data = pd.read_csv('./dataset/classification/test_set.csv')
        tr_sents, tr_labels = train_data['sentence'], train_data['label']
        ts_sents = test_data['sentence']

        # tokenization
        tr_tokens = tokenization(tr_sents)
        ts_tokens = tokenization(ts_sents)
        if isDebug: print(f'tr_sents:{tr_sents[:5]}, tr_labels: {tr_labels[:5]}')
        if isDebug: print(f'ts_sents:{ts_sents[:5]}')
        if isDebug: print(f'tr_tokens: {tr_tokens[:5]}\nts_tokens:{ts_tokens[:5]}')

        # padding & truncation
        max_length = 20
        tr_tokens, _ = pad_sequence(tr_tokens, max_length)
        ts_tokens, _ = pad_sequence(ts_tokens, max_length)
        if isDebug: print(f'padded_tr_tokens:{tr_tokens[:5]}\nts_tokens:{ts_tokens[:5]}')

        with open('./dataset/classification/glove_word.json', "r") as f:
            glove_word = json.load(f)
        word_dict = {}
        word_embedding_matrix = []
        for i, k in enumerate(glove_word.keys()):
            word_dict[k] = i
            word_embedding_matrix.append(torch.tensor(glove_word[k], requires_grad=False))
        word_embedding_matrix = torch.stack(word_embedding_matrix, dim=0)
        if isDebug: print(word_embedding_matrix.shape)
        if isDebug: print(word_embedding_matrix[0])

        train_x = word2vec(tr_tokens, word_dict, word_embedding_matrix)
        test_x = word2vec(ts_tokens, word_dict, word_embedding_matrix)
        if isDebug: print(train_x.shape, test_x.shape)
        train_y = torch.tensor(tr_labels)

        num_classes = 6
        net = model.RNN(args, num_classes).to(device)
        best_dev_acc = [-1]
        best_weights = [None]

        if args.do_train:
            for epoch in range(args.epochs):
                tr_loss, tr_acc = classification_train(train_x, train_y, epoch)
                ts_loss, ts_acc = classification_test(train_x, train_y)
                print("loss: {:.4f}, acc: {:.4f} ts_loss: {:.4f}, ts_acc: {:.4f}".format(tr_loss, tr_acc, ts_loss, ts_acc))
                if ts_acc > best_dev_acc[0]:
                    if hasattr(net, "module"):
                        best_weights[0] = {k: v.to("cpu").clone() for k, v in net.module.state_dict().items()}
                    else:
                        best_weights[0] = {k: v.to("cpu").clone() for k, v in net.state_dict().items()}
                    best_dev_acc[0] = ts_acc
            torch.save(best_weights[0], "pytorch_classification_model.bin")
        else:
            print('do evaluation')
            net.load_state_dict(torch.load('pytorch_classification_model.bin', map_location="cpu"))
            net.to(device)
            pred_y = classification_test(test_x)
            output2csv(pred_y)

    # pos-tagging
    elif args.task == 'pos':
        # load data
        with open('./dataset/pos/train_set.json', 'r') as f:
            train_data_dict = json.load(f)
        with open('./dataset/pos/test_set.json', 'r') as f:
            test_data_dict = json.load(f)
        with open('./dataset/pos/tgt.txt', 'r') as f:
            tgt = f.read()
        tgt = tgt.split()
        tgt_dict = {v: t for (t, v) in enumerate(tgt)}
        if isDebug: print(f'tgt_dict: {tgt_dict}')
        train_data = []
        train_label = []
        test_data = []
        for k in train_data_dict.keys():
            train_data.append(train_data_dict[k]['tokens'])
            train_label.append(train_data_dict[k]['ud_tags'])

        for k in test_data_dict.keys():
            test_data.append(test_data_dict[k]['tokens'])

        if isDebug: print(train_data[0], train_label[0])
        train_max_length = 20

        reverse_train_data = flip_input(train_data, train_max_length)
        tr_x_tokens, tr_x_pad_start_index = pad_sequence(train_data, train_max_length, False)
        reverse_tr_x_tokens, _ = pad_sequence(reverse_train_data, train_max_length, False)
        tr_y_tokens, _ = pad_sequence(train_label, train_max_length, False)

        test_max_length = max(len(x) for x in test_data)
        if isDebug: print(f'test_max_length: {test_max_length}')
        reverse_test_data = flip_input(test_data, test_max_length)
        ts_x_tokens, ts_x_pad_start_index = pad_sequence(test_data, test_max_length, False)
        reverse_ts_x_tokens, _ = pad_sequence(reverse_test_data, test_max_length, False)

        if isDebug: print(f'tr_x_tokens[0]: {tr_x_tokens[0]}, reverse_tr_x_tokens[0]: {reverse_tr_x_tokens[0]}')
        if isDebug: print(f'ts_x_tokens[0]: {ts_x_tokens[0]}, reverse_ts_x_tokens[0]: {reverse_ts_x_tokens[0]}')
        with open('./dataset/pos/fasttext_word.json', "r") as f:
            fasttext_word = json.load(f)
        word_dict = {}
        word_embedding_matrix = []
        for i, k in enumerate(fasttext_word.keys()):
            word_dict[k] = i
            word_embedding_matrix.append(torch.tensor(fasttext_word[k], requires_grad=False))
        word_embedding_matrix = torch.stack(word_embedding_matrix, dim=0)
        if isDebug: print(word_embedding_matrix.shape)
        if isDebug: print(word_embedding_matrix[0])
        train_x = word2vec(tr_x_tokens, word_dict, word_embedding_matrix)
        reverse_train_x = word2vec(reverse_tr_x_tokens, word_dict, word_embedding_matrix)

        train_y = []
        for label in tr_y_tokens:
            words = []
            for w in label:
                words.append(tgt_dict[w])
            train_y.append(words)
        train_y = torch.tensor(train_y)

        test_x = word2vec(ts_x_tokens, word_dict, word_embedding_matrix)
        reverse_test_x = word2vec(reverse_ts_x_tokens, word_dict, word_embedding_matrix)
        if isDebug: print(f'train_x.shape: {train_x.shape}, train_y.shape: {train_y.shape}, test_x.shape: {test_x.shape}')
        if isDebug: print(f'train_x[0]: {train_x[0]}, train_y[0]: {train_y[0]}')

        num_classes = 18
        net = model.BidirectionalRNN(args, num_classes).to(device)
        best_dev_acc = [-1]
        best_weights = [None]

        if args.do_train:
            for epoch in range(args.epochs):
                tr_loss, tr_acc = pos_train(train_x, reverse_train_x, tr_x_pad_start_index, train_max_length, train_y, epoch)
                ts_loss, ts_acc = pos_test(train_x, reverse_train_x, tr_x_pad_start_index, train_max_length, train_y)
                print("loss: {:.4f}, acc: {:.4f} ts_loss: {:.4f}, ts_acc: {:.4f}".format(tr_loss, tr_acc, ts_loss, ts_acc))
                if ts_acc > best_dev_acc[0]:
                    if hasattr(net, "module"):
                        best_weights[0] = {k: v.to("cpu").clone() for k, v in net.module.state_dict().items()}
                    else:
                        best_weights[0] = {k: v.to("cpu").clone() for k, v in net.state_dict().items()}
                    best_dev_acc[0] = ts_acc
            torch.save(best_weights[0], "pytorch_pos_masked_model.bin")
        else:
            print('do evaluation')
            net.load_state_dict(torch.load('pytorch_pos_model.bin', map_location="cpu"))
            net.to(device)
            print(test_x.shape, reverse_test_x.shape, len(ts_x_pad_start_index))
            pred_y = pos_test(test_x, reverse_test_x, ts_x_pad_start_index, test_max_length)
            output2csv(pred_y, 'pos_class.pred.csv')