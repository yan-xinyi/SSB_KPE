# -*- coding: utf-8 -*-
'''
    This file is mainly used to read data, traversal according to different datasets. 
    To run this file, please run main() directly.
'''

import json
import os
from tqdm import tqdm
from nltk import sent_tokenize, word_tokenize
from config import punctuations, stemmer, corpus_list, \
    corpus_field_list_dict, root_dataset_path
from utils import label_keyword, bilstmcrf_get_targets

# Data preprocessing
def data_processed(data_path, fields):

    # Read data
    documents = json.load(open(data_path, 'r', encoding='utf-8'))
    targets = bilstmcrf_get_targets(data_path)

    rv_datas = []
    with tqdm(documents) as pbar:
        for index, document in enumerate(pbar):
            text = []
            for field in fields:
                text.append(document[field])
            text = ". ".join(text).lower().strip()
            # Sentence segmentation
            sentences = sent_tokenize(text)
            # Author keywords
            target = sorted(targets[index], key=lambda item: len(word_tokenize(item)), reverse=False)
            # Traverse sentences
            sen_datas = []
            for sentence in sentences:
                # Word segmentation
                seg_words = [word for word in word_tokenize(sentence) if word not in punctuations]
                stem_words = [stemmer.stem(word) for word in seg_words]
                if len(seg_words) == 0: continue
                # Label words
                _, word_labels = label_keyword(target, stem_words)
                assert len(seg_words) == len(word_labels)
                item = (seg_words, word_labels)
                sen_datas.append(item)
            rv_datas.append(sen_datas)
            pbar.set_description("processing")
    return rv_datas


# Main function
def main():

    # Traversal according to different datasets
    for corpus in corpus_list:
        print('=' * 30, corpus, '=' * 30)
        field_list = corpus_field_list_dict[corpus]
        for field in field_list:
            fields = ['te', field]
            names = "-".join(fields)
            print("fields:", fields)
            documents = {'train': [], 'val':[], 'test': []}
            for dataType in ['train', 'val', 'test']:
                data_path = './inputs/%s/%s.json'%(corpus, dataType) if dataType != 'test' else \
                    os.path.join(root_dataset_path, corpus, 'test.json')
                rv_datas = data_processed(data_path,  fields)
                documents[dataType] = rv_datas
            vocab_list = []
            for samples in tqdm(documents['train']):
                for sample in samples:
                    vocab_list.extend(sample[0])
            for samples in tqdm(documents['val']):
                for sample in samples:
                    vocab_list.extend(sample[0])
            for samples in tqdm(documents['test']):
                for sample in samples:
                    vocab_list.extend(sample[0])
            datas = {
                "documents": documents,
                "vocab_list": ["[PAD]", "[UNK]"] + list(set(vocab_list)),
            }
            json.dump(datas, open('./inputs/%s/%s.json'%(corpus, names), 'w', encoding='utf-8'))


if __name__ == '__main__':

    main()
