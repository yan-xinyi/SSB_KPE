# -*- coding: utf-8 -*-

import random
import json
import os
import numpy as np
import pandas as pd
import torch
from nltk import sent_tokenize, word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import BertTokenizer
from configs import LABELS2IDS, BERT_PATH
from torch.backends import cudnn

# Set random seed for reproducibility
def set_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = '0'
    cudnn.benchmark = False
    cudnn.deterministic = True

# Calculate precision, recall, and F1-score metrics
def metrics(y_true, y_pred):
    p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return round(p*100, 2), round(r*100, 2), round(f1*100, 2)

# Compute evaluation metrics and split results by dataset
def compute_metrics(input_logits, labels, dataset_ids = None, is_output_dataset_results=False):
    # 处理输入
    y_pred = torch.argmax(torch.softmax(input_logits, dim=-1), dim=-1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    if is_output_dataset_results:

        # Split results by dataset (PMC, LIS, IEEE)
        pmc_y_true, pmc_y_pred = [], []
        lis_y_true, lis_y_pred = [], []
        ieee_y_true, ieee_y_pred = [], []
        for _label, _pred, _dataset_id in zip(y_true, y_pred, dataset_ids):
            if _dataset_id == 1:
                pmc_y_true.append(_label)
                pmc_y_pred.append(_pred)
            elif _dataset_id == 2:
                lis_y_true.append(_label)
                lis_y_pred.append(_pred)
            elif _dataset_id == 3:
                ieee_y_true.append(_label)
                ieee_y_pred.append(_pred)

        # Compute metrics for each dataset and overall results
        pmc_p, pmc_r, pmc_f1 = metrics(pmc_y_true, pmc_y_pred)
        lis_p, lis_r, lis_f1 = metrics(lis_y_true, lis_y_pred)
        ieee_p, ieee_r, ieee_f1 = metrics(ieee_y_true, ieee_y_pred)
        all_p, all_r, all_f1 = metrics(y_true, y_pred)

        return (pmc_p, pmc_r, pmc_f1), (lis_p, lis_r, lis_f1), \
               (ieee_p, ieee_r, ieee_f1), (all_p, all_r, all_f1)
    else:
        # Compute overall results without dataset splitting
        all_p, all_r, all_f1 = metrics(y_true, y_pred)
        return all_p, all_r, all_f1


# Function to encode data (sections or sentences) into token IDs
def encoder_datas(datas, sec_name_len=32, sec_header_len=128,
                sec_text_len=2048, max_sen_nums=80, max_sen_len=64,
                input_text_type='section', save_fold='./inputs/input_datas'):

    file_name, file_path = '', ''
    if input_text_type=='section':
        file_name = "%s-%s-%s-%s"%(input_text_type, sec_name_len,
                                   sec_header_len, sec_text_len)
        file_path = os.path.join(save_fold, file_name)
    elif input_text_type == 'sentence':
        file_name = "%s-%s-%s-%s-%s" % (input_text_type, sec_name_len,
                                        sec_header_len, max_sen_nums, max_sen_len)
        file_path = os.path.join(save_fold, file_name)
    elif input_text_type=='for-bert-section':
        sec_text_len = 512
        file_name = "%s-%s-%s-%s" % (input_text_type, sec_name_len,
                                     sec_header_len, sec_text_len)
        file_path = os.path.join(save_fold, file_name)
    elif input_text_type=='for-bert-sentence':
        file_name = "%s-%s-%s-%s-%s" % (input_text_type, sec_name_len,
                                        sec_header_len, max_sen_nums, max_sen_len)
        file_path = os.path.join(save_fold, file_name)

    if not os.path.exists(file_path):

        # Tokenizer
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

        # Extract various features from the dataset
        p_ids = datas['p_id'].tolist()
        sec_titles = datas['sec_title'].tolist()
        sec_subtitles = datas['sec_subtitle'].tolist()
        sec_texts = datas['sec_text'].tolist()
        labels = [LABELS2IDS[anno_result] for anno_result in datas['label']]
        dataset_ids = datas['dataset_id'].values.flatten().tolist()
        jname_ids = datas['jname_id'].values.flatten().tolist()

        # Other feature extraction (e.g., reference counts, section lengths, etc.)
        bib_nums = datas['bib_num'].values.flatten().tolist()
        fn_nums = datas['fn_num'].values.flatten().tolist()
        fig_nums = datas['fig_num'].values.flatten().tolist()
        tab_nums = datas['tab_num'].values.flatten().tolist()
        equ_nums = datas['equ_num'].values.flatten().tolist()
        para_nums = datas['para_num'].values.flatten().tolist()
        sen_nums = datas['sen_num'].values.flatten().tolist()
        word_nums = datas['word_num'].values.flatten().tolist()
        sec_locs = datas['sec_loc'].values.flatten().tolist()

        # Initialize empty lists for encoded data
        sec_title_ids = [None] * len(p_ids)
        sec_subtitle_ids = [None] * len(p_ids)
        sec_text_ids = [None] * len(p_ids)

        # Handle different text input types (section, sentence, BERT-based encoding)
        if input_text_type == 'section':
            for item in tqdm(range(len(p_ids))):
                # Encode section titles, subtitles, and texts using BERT tokenizer
                # print(str(sec_titles[item]))
                sec_title_ids[item] = json.dumps(tokenizer.encode(
                    sec_titles[item].lower().strip(),
                    add_special_tokens=False,
                    truncation=True,
                    padding='max_length',
                    max_length=sec_name_len))

                # Encode section titles, subtitles, and texts using BERT tokenizer
                sec_subtitle_ids[item] = json.dumps(tokenizer.encode(
                    sec_subtitles[item].lower().strip(),
                    add_special_tokens=False,
                    truncation=True,
                    padding='max_length',
                    max_length=sec_header_len))

                sec_text_id = tokenizer.encode(sec_texts[item].lower().strip(), add_special_tokens=False)
                sec_text_id_len = len(sec_text_id)
                half_sec_text_len = sec_text_len // 2
                # If the text is too long, take the first and last parts of the section text
                if sec_text_id_len > sec_text_len:
                    forward = sec_text_id[0: half_sec_text_len]
                    backward = sec_text_id[-half_sec_text_len: ]
                    sec_text_ids[item] = json.dumps(forward + backward)
                else:
                    padding = [0] * (sec_text_len - sec_text_id_len)
                    sec_text_id += padding
                    sec_text_ids[item] = json.dumps(sec_text_id)

        # For other input types (such as BERT-based encoding or sentence encoding), similar processing is done
        elif input_text_type == 'for-bert-section':
            sec_text_len = 512
            for item in tqdm(range(len(p_ids))):

                # Encode section titles, subtitles, and texts using BERT tokenizer
                sec_title_ids[item] = json.dumps(tokenizer.encode(
                    sec_titles[item].lower().strip(),
                    add_special_tokens=False,
                    truncation=True,
                    padding='max_length',
                    max_length=sec_name_len))

                sec_subtitle_ids[item] = json.dumps(tokenizer.encode(
                    sec_subtitles[item].lower().strip(),
                    add_special_tokens=False,
                    truncation=True,
                    padding='max_length',
                    max_length=sec_header_len))

                sec_text_id = tokenizer.encode(sec_texts[item].lower().strip())
                sec_text_id_len = len(sec_text_id)
                half_sec_text_len = sec_text_len // 2
                if sec_text_id_len > sec_text_len:
                    forward = sec_text_id[0: half_sec_text_len]
                    backward = sec_text_id[-half_sec_text_len: ]
                    sec_text_ids[item] = json.dumps(forward + backward)
                else:
                    padding = [0] * (sec_text_len - sec_text_id_len)
                    sec_text_id += padding
                    sec_text_ids[item] = json.dumps(sec_text_id)

        elif input_text_type == 'sentence':
            for item in tqdm(range(len(p_ids))):

                # Encode the section title
                sec_title_ids[item] = json.dumps(tokenizer.encode(
                    sec_titles[item].lower().strip(),
                    add_special_tokens=False,
                    truncation=True,
                    padding='max_length',
                    max_length=sec_name_len))

                # Encode the section subtitle
                sec_subtitle_ids[item] = json.dumps(tokenizer.encode(
                    sec_subtitles[item].lower().strip(),
                    add_special_tokens=False,
                    truncation=True,
                    padding='max_length',
                    max_length=sec_header_len))

                # Tokenize the section text into sentences
                sentences = sent_tokenize(sec_texts[item].lower().strip())

                # Current number of sentences in the section
                curr_sen_num = len(sentences)

                # If the number of sentences is within the allowed limit
                if curr_sen_num <= max_sen_nums:
                    _index = 0
                    sec_text_id = np.zeros((max_sen_nums, max_sen_len))
                    for sentence in sentences:
                        if _index > max_sen_nums - 1: break
                        sentence = " ".join([word.strip() for word in word_tokenize(sentence) if len(word.strip()) > 0])
                        sec_text_id[_index] = tokenizer.encode(
                                                 sentence,
                                                 add_special_tokens=False,
                                                 truncation=True,
                                                 padding='max_length',
                                                 max_length=max_sen_len)
                        _index += 1
                    sec_text_ids[item] = json.dumps(sec_text_id.tolist())
                else:
                    # If the number of sentences exceeds the limit, select a portion of sentences
                    half_max_sen_nums = max_sen_nums // 2
                    forward_sentences = sentences[0: half_max_sen_nums]
                    backward_sentences = sentences[-half_max_sen_nums: ]
                    sentences = forward_sentences + backward_sentences

                    _index = 0
                    sec_text_id = np.zeros((max_sen_nums, max_sen_len))
                    for sentence in sentences:
                        if _index > max_sen_nums - 1: break
                        sentence = " ".join([word.strip() for word in word_tokenize(sentence) if len(word.strip()) > 0])
                        sec_text_id[_index] = tokenizer.encode(
                            sentence,
                            add_special_tokens=False,
                            truncation=True,
                            padding='max_length',
                            max_length=max_sen_len)
                        _index += 1
                    sec_text_ids[item] = json.dumps(sec_text_id.tolist())

        elif input_text_type == 'for-bert-sentence':
            for item in tqdm(range(len(p_ids))):

                # Encode the section title
                sec_title_ids[item] = json.dumps(tokenizer.encode(
                    sec_titles[item].lower().strip(),
                    add_special_tokens=False,
                    truncation=True,
                    padding='max_length',
                    max_length=sec_name_len))

                # Encode the section title
                sec_subtitle_ids[item] = json.dumps(tokenizer.encode(
                    sec_subtitles[item].lower().strip(),
                    add_special_tokens=False,
                    truncation=True,
                    padding='max_length',
                    max_length=sec_header_len))

                # Tokenize the section text into sentences
                sentences = sent_tokenize(sec_texts[item].lower().strip())
                # Current number of sentences in the section
                curr_sen_num = len(sentences)
                # If the number of sentences is within the allowed limit
                if curr_sen_num <= max_sen_nums:
                    _index = 0
                    sec_text_id = np.zeros((max_sen_nums, max_sen_len))
                    for sentence in sentences:
                        if _index > max_sen_nums - 1: break
                        sentence = " ".join([word.strip() for word in word_tokenize(sentence) if len(word.strip()) > 0])
                        sec_text_id[_index] = tokenizer.encode(
                            sentence,
                            truncation=True,
                            padding='max_length',
                            max_length=max_sen_len)
                        _index += 1
                    sec_text_ids[item] = json.dumps(sec_text_id.tolist())
                else:
                    # If the number of sentences exceeds the limit, select a portion of sentences
                    half_max_sen_nums = max_sen_nums // 2
                    forward_sentences = sentences[0: half_max_sen_nums]
                    backward_sentences = sentences[-half_max_sen_nums:]
                    sentences = forward_sentences + backward_sentences

                    _index = 0
                    sec_text_id = np.zeros((max_sen_nums, max_sen_len))
                    for sentence in sentences:
                        if _index > max_sen_nums - 1: break
                        sentence = " ".join([word.strip() for word in word_tokenize(sentence) if len(word.strip()) > 0])
                        sec_text_id[_index] = tokenizer.encode(
                            sentence,
                            truncation=True,
                            padding='max_length',
                            max_length=max_sen_len)
                        _index += 1
                    sec_text_ids[item] = json.dumps(sec_text_id.tolist())


        # Convert the encoded data into a pandas DataFrame
        df = pd.DataFrame(np.array(
            [p_ids, labels,
             dataset_ids, jname_ids,
             bib_nums, fn_nums, fig_nums, tab_nums, equ_nums,
             para_nums, sen_nums, word_nums, sec_locs,
             sec_title_ids, sec_subtitle_ids, sec_text_ids]).T,
             columns=['p_id', 'label', 'dataset_id', 'jname_id',
                 'bib_num', 'fn_num', 'fig_num', 'tab_num',
                 'equ_num', 'para_num', 'sen_num', 'word_num', 'sec_loc',
                 'sec_title_id', 'sec_subtitle_id', 'sec_text_id'])

        # Save the data to a CSV file
        df.to_csv(file_path, sep='\t', header=False, index=False)

        return df, file_name
    else:
        # If the file already exists, read the data from the CSV file
        df = pd.read_csv(file_path, sep='\t', names=
                ['p_id', 'label', 'dataset_id', 'jname_id',
                 'bib_num', 'fn_num', 'fig_num', 'tab_num',
                 'equ_num', 'para_num', 'sen_num', 'word_num', 'sec_loc',
                 'sec_title_id', 'sec_subtitle_id', 'sec_text_id'])

        return df, file_name