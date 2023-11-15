# -*- coding: utf-8 -*-
'''
    Split a part of the data from the training set as a validation set.
'''

import json
import os
from config import corpus_list, root_dataset_path

def main():

    for corpus in corpus_list:

        print("DATASET:", corpus)
        data_indexs = json.load(open(
            os.path.join(root_dataset_path, corpus, 't_v_index.json'), 'r', encoding='utf-8'))
        datas = json.load(open(
            os.path.join(root_dataset_path, corpus, 'train.json'), 'r', encoding='utf-8'))
        train_indexs = data_indexs['train_indexs']
        val_indexs = data_indexs['val_indexs']

        train_list, val_list = [], []
        for index in train_indexs: train_list.append(datas[index])
        for index in val_indexs: val_list.append(datas[index])
        print("Train size:", len(train_list))
        print("Val size:", len(val_list))

        # save to file
        json.dump(train_list, open(
            os.path.join('./inputs', corpus, 'train.json'), 'w', encoding='utf-8'), ensure_ascii=False)
        json.dump(val_list, open(
            os.path.join('./inputs', corpus, 'val.json'), 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':

    main()