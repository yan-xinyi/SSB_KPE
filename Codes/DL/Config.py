# -*- coding: utf-8 -*-
'''
This python file is mainly used for hyperparameter setting. As shown in the following parameter items, it mainly includes:
'''

import torch
from nltk.corpus import stopwords
from nltk import PorterStemmer
from transformers import BertTokenizer, T5Tokenizer


root_dataset_path = '../dataset'
bert_path = '../scibert-model'
tag2ids = {"[PAD]": 0, 'K_B': 1, 'K_I': 2, '0': 3}
id2tags = {val: key for key, val in tag2ids.items()}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stops_words = set(stopwords.words("english"))

stemmer = PorterStemmer()

weight = 't5-base'     # 't5-base' for T5-base model, 't5-large' for T5-laerge model
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
tokenizer = T5Tokenizer.from_pretrained(weight)

punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')', '[', ']',
                '&', '!', '*', '@', '#', '$', '%', '\\','\"','}','{', '-',
                '–', '—', '..', '/', '=', '∣', '…', '′', '⋅', '×', '+', '•',
                '<', '>', '’', "''"]
corpus_list = ['corpus-ph', 'corpus-mh', 'corpus-ml']
corpus_field_list_dict = {
    "corpus-ph": ['ft','ib', 'rw', 'ab', 'md',
                'er', 'dc',
                's0', 's1', 's2', 's3', 's4',
                'w0', 'w1', 'w2', 'w3', 'w4',],
    "corpus-mh": ['ft','ib', 'rw', 'ab', 'md',
                'er', 'dc',
                's0', 's1', 's2', 's3', 's4',
                'w0', 'w1', 'w2', 'w3', 'w4'],
    "corpus-ml": ['ft','ib', 'rw', 'ab', 'md',
                'er', 'dc',
                's0', 's1', 's2', 's3', 's4',
                'w0', 'w1', 'w2', 'w3', 'w4']
}

# parameters
embed_dim = 256
hidden_dim = 256
batch_size = 64
max_length = 64
add_bilstm = True

lr1 = 1e-4
lr2 = 2e-5
# max_norm = 0.25
weight_decay = 2e-5
factor = 0.5
patience = 3
epoches = 20
