# -*- coding: utf-8 -*-
'''
   This python file mainly contains code for training, testing and evaluation of KPE models for Bert+BiLSTM+CRF.
   The main function is BERT_BiLSTM_CRF().
      1* Through the SentenceDataSet(Dataset): build the dictionary for vocabulary and cognitive signals.
      2* Build BERT_BiLSTM_CRF() model
      3* Start training and calculate the loss value.
      4* Conduct testing, again read the data first, build the model, in the prediction, evaluate its results.
'''

import time
import copy
import json
import os
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import *
from torchcrf import CRF
from transformers import BertModel, BertConfig
from utils import set_seed, evaluate, bertbilstmcrf_load_datas, \
    bertbilstmcrf_get_targets, bertbilstmcrf_parser_entities

# Fixed random seed
set_seed(2022)

torch.cuda.set_device(0)

class SentenceDataSet(Dataset):

    def __init__(self, samples, max_length=64):
        self.datas = samples
        self.max_length = max_length

    def __getitem__(self, item):
        item = self.datas[item]
        input_ids = item['input_ids']
        token_type_ids = item['token_type_ids']
        attention_mask = item['attention_mask']
        tags = item['tags']
        seq_length = len(input_ids)
        # Truncate data
        if seq_length < self.max_length:
            zero_pad = [0] * (self.max_length - seq_length)
            one_pad = [1] * (self.max_length - seq_length)
            input_ids.extend(zero_pad)
            token_type_ids.extend(one_pad)
            attention_mask.extend(zero_pad)
            tags.extend(zero_pad)
        else:
            input_ids = input_ids[: self.max_length]
            token_type_ids = token_type_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            tags = tags[: self.max_length]

        one_data = {
            "input_ids": torch.tensor(input_ids).long().to(device),
            "token_type_ids": torch.tensor(token_type_ids).long().to(device),
            "attention_mask": torch.tensor(attention_mask).long().to(device),
            "tags": torch.tensor(tags).long().to(device),
            "seq_length": torch.tensor(seq_length).long().to(device)
        }
        return one_data

    def __len__(self):
        return len(self.datas)

# Data generation
def yield_data(samples=None, batch_size=8, max_length=64, shuffle=False, is_pos_samples=True):
    # Processing data
    processd_datas, sec_infos = bertbilstmcrf_load_datas(samples=samples,
                                is_pos_samples=is_pos_samples)
    tmp = SentenceDataSet(samples=processd_datas, max_length=max_length)
    return DataLoader(tmp, batch_size=batch_size, shuffle=shuffle), sec_infos

# BERT_BiLSTM-CRF
class BERT_BiLSTM_CRF(nn.Module):

    def __init__(self, hidden_dim=256, num_tags=4, dropout=0.5,
                  num_layers=1, add_bilstm=False):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.add_bilstm = add_bilstm
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_config = BertConfig.from_pretrained(bert_path)
        if add_bilstm:
            self.bilstm = nn.LSTM(input_size=self.bert_config.hidden_size,
                                  hidden_size=hidden_dim, num_layers=num_layers,
                                  bidirectional=True, batch_first=True)
            self.layernorm = nn.LayerNorm(normalized_shape=2 * hidden_dim)
            self.tocrf = nn.Linear(2 * hidden_dim, num_tags)
        else:
            self.layernorm = nn.LayerNorm(normalized_shape=self.bert_config.hidden_size)
            self.tocrf = nn.Linear(self.bert_config.hidden_size, num_tags)
        self.crf = CRF(num_tags=num_tags, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs, is_training=True):

        input = self.bert(input_ids=inputs['input_ids'],
                          attention_mask=inputs['attention_mask'],
                          token_type_ids=inputs['token_type_ids'])
        if self.add_bilstm:
            input = self.bilstm(input[0])
        input = self.dropout(torch.relu(self.layernorm(input[0])))
        crf_feats = self.tocrf(input)
        if is_training:
            loss = self.crf(emissions=crf_feats,
                            tags=inputs['tags'],
                            mask=inputs['attention_mask'].bool(),
                            reduction='mean')
            return -loss
        else:
            outputs = self.crf.decode(emissions=crf_feats, mask=inputs['attention_mask'].bool())
            tag_probs = torch.softmax(crf_feats, dim=-1)
            return tag_probs, outputs

# Training model
def train(data_path, val_data_path):

    # Read data
    datas = json.load(open(data_path, 'r', encoding='utf-8'))
    train_documents, val_documtens = datas['documents']['train'], datas['documents']['val']
    targets = bertbilstmcrf_get_targets(val_data_path)
    # Load data
    train_dataloader, _ = yield_data(samples=train_documents, shuffle=True,
                        batch_size=batch_size, is_pos_samples=True, max_length=max_length)
    val_dataloader, sec_infos = yield_data(samples=val_documtens,
                        batch_size=batch_size, is_pos_samples=False, max_length=max_length)

    # Model and optimizer
    model = BERT_BiLSTM_CRF(add_bilstm=add_bilstm, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr2, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor,
                                                     patience=patience, verbose=True)
    print(model)
    best_f1, last_f1, best_model = 0, 0, None
    for epoch in range(epoches):
        model.train()
        losses = []
        with tqdm(train_dataloader) as pbar_train:
            for batch_train, inputs in enumerate(pbar_train):
                loss = model(inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar_train.set_description("Epoch: %s, loss: %s"%(epoch+1, round(loss.item(), 3)))
        print("Loss: %s"%(np.average(losses)))
        scheduler.step(np.average(losses))
        rv_datas = {
            "input_ids": [],
            "tag_probs": [],
            "outputs": []
        }
        model.eval()
        with tqdm(val_dataloader) as pbar_val:
            for rank, inputs in enumerate(pbar_val):
                tag_probs, outputs = model(inputs, is_training=False)
                tag_probs = tag_probs.detach().cpu().numpy().tolist()
                input_ids = inputs['input_ids'].detach().cpu().numpy().tolist()
                rv_datas["input_ids"].extend(input_ids)
                rv_datas["tag_probs"].extend(tag_probs)
                rv_datas["outputs"].extend(outputs)
                pbar_val.set_description("Predict")

        # Get results
        preds, proba_preds = [], []
        for sec_info in sec_infos:
            start, end = sec_info[0], sec_info[1]
            input_ids = rv_datas["input_ids"][start: end]
            tag_probs = rv_datas["tag_probs"][start: end]
            outputs = rv_datas["outputs"][start: end]
            proba_preds_list = []
            for batch in range(len(outputs)):
                probs = tag_probs[batch]  # seq, tag_size
                tags = [id2tags[tag] for tag in outputs[batch]]  # seq
                words = [tokenizer.decode([input_id]) for input_id in input_ids[batch]]
                prob_list = []
                for index in range(len(tags)):
                    prob = probs[index]
                    temp = {}
                    for _i in range(len(prob)):
                        temp[id2tags[_i]] = prob[_i]
                    prob_list.append(temp)
                result = bertbilstmcrf_parser_entities(words, tags, probs=prob_list)
                proba_pred = sorted(result, key=lambda item: item[1], reverse=True)
                proba_preds_list.extend(proba_pred)

            # Get the extraction keywords for each document
            kw_dict = {}
            for result in proba_preds_list:
                word = result[0]
                prob = result[1]
                if word not in kw_dict.keys():
                    kw_dict[word] = [prob]
                else:
                    kw_dict[word].append(prob)
            kw_dict = [[key, np.average(val)] for key, val in kw_dict.items()]
            pred = [item[0] for item in kw_dict]
            proba_pred = sorted(kw_dict, key=lambda item: item[1], reverse=True)

            # Store the results
            preds.append(pred)
            proba_preds.append(proba_pred)

        # Result evaluation
        result = evaluate(preds, targets, topk=-1)

        print(result)
        if result[2] > best_f1:
            best_f1 = result[2]
            best_model = copy.deepcopy(model)
        print('='*100)

        # Early stop
        if epoch > 5 and (result[2] == 0.0 or (last_f1 - result[2] >= 10)):
            return best_model
        last_f1 = result[2]

    return best_model

# Result prediction
def predict(data_path, model):

    # Read data
    datas = json.load(open(data_path, 'r', encoding='utf-8'))
    test_documtens = datas['test']

    # Load data
    test_dataloader, sec_infos = yield_data(samples=test_documtens,
                     batch_size=batch_size, is_pos_samples=False, max_length=max_length)
    rv_datas = {
        "input_ids": [],
        "tag_probs": [],
        "outputs": []
    }
    model.eval()
    with tqdm(test_dataloader) as pbar_test:
        for rank, inputs in enumerate(pbar_test):
            tag_probs, outputs = model(inputs, is_training=False)
            tag_probs = tag_probs.detach().cpu().numpy().tolist()
            input_ids = inputs['input_ids'].detach().cpu().numpy().tolist()
            rv_datas["input_ids"].extend(input_ids)
            rv_datas["tag_probs"].extend(tag_probs)
            rv_datas["outputs"].extend(outputs)
            pbar_test.set_description("Predict")

    # Get results
    preds, proba_preds = [], []
    for sec_info in sec_infos:
        start, end = sec_info[0], sec_info[1]
        input_ids = rv_datas["input_ids"][start: end]
        tag_probs = rv_datas["tag_probs"][start: end]
        outputs = rv_datas["outputs"][start: end]
        proba_preds_list = []
        for batch in range(len(outputs)):
            probs = tag_probs[batch]  # seq, tag_size
            tags = [id2tags[tag] for tag in outputs[batch]]  # seq
            words = [tokenizer.decode([input_id]) for input_id in input_ids[batch]]
            prob_list = []
            for index in range(len(tags)):
                prob = probs[index]
                temp = {}
                for _i in range(len(prob)):
                    temp[id2tags[_i]] = prob[_i]
                prob_list.append(temp)
            result = bertbilstmcrf_parser_entities(words, tags, probs=prob_list)
            proba_pred = sorted(result, key=lambda item: item[1], reverse=True)
            proba_preds_list.extend(proba_pred)

        # Get the extraction keywords for each document
        kw_dict = {}
        for result in proba_preds_list:
            word = result[0]
            prob = result[1]
            if word not in kw_dict.keys():
                kw_dict[word] = [prob]
            else:
                kw_dict[word].append(prob)
        kw_dict = [[key, np.average(val)] for key, val in kw_dict.items()]
        pred = [item[0] for item in kw_dict]
        proba_pred = sorted(kw_dict, key=lambda item: item[1], reverse=True)

        # Store the results
        preds.append(pred)
        proba_preds.append(proba_pred)

    return preds, proba_preds


if __name__ == '__main__':

    # Traversal according to different datasets
    for corpus in corpus_list:

        print('=' * 30, corpus, '=' * 30)

        # Test Data Path
        test_data_path = os.path.join(root_dataset_path, corpus, 'test.json')
        # Val Data Path
        val_data_path = './inputs/%s/val.json' % corpus

        # Save path
        save_fold = './inputs/%s/%s'%(corpus, 'bertbilstmcrf')

        # Keyword extraction according to different inputs
        time_info = {}
        field_list = corpus_field_list_dict[corpus]
        for field in field_list:

            fields = ['te', field]
            print('-' * 10, fields, '-' * 10)
            names = '-'.join(fields)

            # Training model
            data_path = './inputs/%s/%s.json' % (corpus, names)
            s_time = time.time()
            model = train(data_path, val_data_path)
            train_time = time.time() - s_time
            print("training time:", train_time)

            # Predict
            s_time = time.time()
            preds, proba_preds = predict(data_path, model)
            test_time = time.time() - s_time
            print("testing time:", test_time)
            time_info[names] = [train_time, test_time]

            # Save to file
            json.dump(proba_preds, open(
                os.path.join(save_fold, 'pred-%s.json'% names), 'w', encoding='utf-8'))

            # Result evaluation
            targets = bertbilstmcrf_get_targets(test_data_path)
            for topk in [3, 5, 10]:
                s = evaluate(preds, targets, topk=topk)
                print(s)

        # Save the time consumed information
        json.dump(time_info, open(
            os.path.join(save_fold, 'time_info.json'), 'w', encoding='utf-8'))

