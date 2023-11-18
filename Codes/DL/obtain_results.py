# -*- coding: utf-8 -*-
'''
    The main function of this document is to integrate the keyword results extracted from each chapter structure into a keyword collection.
'''

import json
import os
import numpy as np
import pandas as pd
from config import corpus_list, root_dataset_path
from utils import evaluate
from utils import bilstmcrf_get_targets, bertbilstmcrf_get_targets
from utils import compute_statistical_significance, integration_results

# Main function
def main(model_name="bilstmcrf", corpus=None, field_list=None, RI_field_lists=None, en_topk=-1):

    data_path = os.path.join(root_dataset_path, corpus, 'test.json')

    input_fold = './inputs/%s/%s' % (corpus, model_name)
    save_fold = './outputs/%s/%s' % (corpus, model_name)

    # Target values
    targets = bilstmcrf_get_targets(data_path) if \
        model_name == 'bilstmcrf' else bertbilstmcrf_get_targets(data_path)

    # Raw data
    documents = json.load(open(data_path, 'r', encoding='utf-8'))

    # Data type
    dns = [document['dn'] for document in documents]

    # Evaluate
    TAB = [[], [], []]
    TFT = [[], [], []]
    results = np.zeros((len(field_list)+len(RI_field_lists), 9))
    for row, field in enumerate(field_list):

        fields = ['te', field]
        print('-' * 30, fields, '-' * 30)
        names = '-'.join(fields)

        # Predict results
        preds_list = json.load(open(os.path.join(
            input_fold, 'pred-%s.json' % names), 'r', encoding='utf-8'))

        preds = []
        for result in preds_list:
            preds.append([item[0] for item in result])

        # Evaluate by dataset
        pmc_preds, lis_preds, ieee_preds = [], [], []
        pmc_trues, lis_trues, ieee_trues = [], [], []
        for true, pred, dn in zip(targets, preds, dns):
            if dn == 'PMC':
                pmc_trues.append(true)
                pmc_preds.append(pred)
            elif dn == 'LIS':
                lis_trues.append(true)
                lis_preds.append(pred)
            elif dn == 'IEEE':
                ieee_trues.append(true)
                ieee_preds.append(pred)

        if field == 'ab':
            TAB[0] = [pmc_preds, pmc_trues]
            TAB[1] = [lis_preds, lis_trues]
            TAB[2] = [ieee_preds, ieee_trues]
        if field == 'ft':
            TFT[0] = [pmc_preds, pmc_trues]
            TFT[1] = [lis_preds, lis_trues]
            TFT[2] = [ieee_preds, ieee_trues]

        # PMC-1316
        print('DATASET:', 'PMC-1316')
        for col, topk in enumerate([3, 5, 10]):
            rs = evaluate(pmc_preds, pmc_trues, topk=topk)
            results[row][col] = rs[2]

            if field not in ['ab', 'ft']:
                if rs[2] > results[0][col]:
                    state1 = compute_statistical_significance(TAB[0], [pmc_preds, pmc_trues], len(pmc_preds), topk=topk)
                    print("TA:", state1)
                if rs[2] > results[1][col]:
                    state2 = compute_statistical_significance(TFT[0], [pmc_preds, pmc_trues], len(pmc_preds), topk=topk)
                    print("TF:", state2)
            print(rs)

        # LIS-2000
        print('DATASET:', 'LIS-2000')
        for col, topk in enumerate([3, 5, 10]):
            rs = evaluate(lis_preds, lis_trues, topk=topk)
            results[row][col + 3] = rs[2]

            if field not in ['ab', 'ft']:
                if rs[2] > results[0][col + 3]:
                    state1 = compute_statistical_significance(TAB[1], [lis_preds, lis_trues], len(lis_preds), topk=topk)
                    print("TA:", state1)
                if rs[2] > results[1][col + 3]:
                    state2 = compute_statistical_significance(TFT[1], [lis_preds, lis_trues], len(lis_preds), topk=topk)
                    print("TF:", state2)
            print(rs)

        # IEEE-2000
        print('DATASET:', 'IEEE-2000')
        for col, topk in enumerate([3, 5, 10]):
            rs = evaluate(ieee_preds, ieee_trues, topk=topk)
            results[row][col + 6] = rs[2]

            if field not in ['ab', 'ft']:
                if rs[2] > results[0][col + 6]:
                    state1 = compute_statistical_significance(TAB[2], [ieee_preds, ieee_trues], len(ieee_preds), topk=topk)
                    print("TA:", state1)
                if rs[2] > results[1][col + 6]:
                    state2 = compute_statistical_significance(TFT[2], [ieee_preds, ieee_trues], len(ieee_preds), topk=topk)
                    print("TF:", state2)
            print(rs)

    # Integration
    print('=' * 30, 'Result Integration', '=' * 30)

    for en_index, RI_field_list in enumerate(RI_field_lists):

        print('=' * 15, RI_field_list, '=' * 15)

        # Fields
        fields_list = []
        for field in RI_field_list:
            fields = ['te', field]
            fields_list.append(fields)

        preds = integration_results(input_fold, fields_list, topk=en_topk, prefix='pred-')

        # Evaluate by dataset
        pmc_preds, lis_preds, ieee_preds = [], [], []
        pmc_trues, lis_trues, ieee_trues = [], [], []
        for true, pred, dn in zip(targets, preds, dns):
            if dn == 'PMC':
                pmc_trues.append(true)
                pmc_preds.append(pred)
            elif dn == 'LIS':
                lis_trues.append(true)
                lis_preds.append(pred)
            elif dn == 'IEEE':
                ieee_trues.append(true)
                ieee_preds.append(pred)

        # PMC-1316
        print('DATASET:', 'PMC-1316')
        for col, topk in enumerate([3, 5, 10]):
            rs = evaluate(pmc_preds, pmc_trues, topk=topk)
            results[len(field_list)+en_index][col] = rs[2]
            if rs[2] > results[0][col]:
                state1 = compute_statistical_significance(TAB[0], [pmc_preds, pmc_trues], len(pmc_preds), topk=topk)
                print("TA:", state1)
            if rs[2] > results[1][col]:
                state2 = compute_statistical_significance(TFT[0], [pmc_preds, pmc_trues], len(pmc_preds), topk=topk)
                print("TF:", state2)
            print(rs)

        # LIS-2000
        print('DATASET:', 'LIS-2000')
        for col, topk in enumerate([3, 5, 10]):
            rs = evaluate(lis_preds, lis_trues, topk=topk)
            results[len(field_list)+en_index][col+3] = rs[2]
            if rs[2] > results[0][col+3]:
                state1 = compute_statistical_significance(TAB[1], [lis_preds, lis_trues], len(lis_preds), topk=topk)
                print("TA:", state1)
            if rs[2] > results[1][col+3]:
                state2 = compute_statistical_significance(TFT[1], [lis_preds, lis_trues], len(lis_preds), topk=topk)
                print("TF:", state2)
            print(rs)

        # IEEE-2000
        print('DATASET:', 'IEEE-2000')
        for col, topk in enumerate([3, 5, 10]):
            rs = evaluate(ieee_preds, ieee_trues, topk=topk)
            results[len(field_list)+en_index][col+6] = rs[2]
            if rs[2] > results[0][col+6]:
                state1 = compute_statistical_significance(TAB[2], [ieee_preds, ieee_trues], len(ieee_preds), topk=topk)
                print("TA:", state1)
            if rs[2] > results[1][col+6]:
                state2 = compute_statistical_significance(TFT[2], [ieee_preds, ieee_trues], len(ieee_preds), topk=topk)
                print("TF:", state2)
            print(rs)

    # Save to file
    if corpus == "corpus-ph":
        pd.DataFrame(results, columns=None,
                     index=['AB', 'FT', 'IN', 'RW', 'MD', 'ER', 'DC',
                           'TSre', 'TStr', 'TSex', 'TSge',
                           'ASre', 'AStr', 'ASex', 'ASge',
                           'zhang', 'nguyen', 'WS', 'SS',
                           'CS', 'AB+TSre', 'AB+TStr', 'AB+TSex',
                           'AB+TSge', 'CS+TSre', 'CS+TStr', 'CS+TSex',
                           'CS+TSge', 'CSre']).to_csv(
            os.path.join(save_fold, 'results.csv'), header=False)
    else:
        pd.DataFrame(results, columns=None,
                     index=['AB', 'FT', 'IN', 'RW', 'MD', 'ER', 'DC',
                            'WS', 'SS', 'CS']).to_csv(
            os.path.join(save_fold, 'results.csv'), header=False)
    print(results)


if __name__ == '__main__':

    model_names = ["bilstmcrf", "bertbilstmcrf"]
    field_list_dict = {"corpus-ph": ['ab', 'ft',
                                     'ib', 'rw',
                                     'md', 'er', 'dc',
                                     'TS(Re)-256', 'TS(Tr)-256',
                                     'TS(Ex)-256', 'TS(Ge)-256',
                                     'AS(Re)-256', 'AS(Tr)-256',
                                     'AS(Ex)-256', 'AS(Ge)-256',
                                     'nguyen', 'zhang'],
                       "corpus-mh": ['ab', 'ft',
                                     'ib', 'rw', 'md', 'er', 'dc'],
                       "corpus-ml": ['ab', 'ft',
                                     'ib', 'rw', 'md', 'er', 'dc'],
                       }
    RI_field_lists_dict = {"corpus-ph": [['w0', 'w1', 'w2', 'w3', 'w4'],
                                         ['s0', 's1', 's2', 's3', 's4'],
                                         ['ib', 'rw', 'md', 'er', 'dc'],
                                         ['ab', 'TS(Re)-256'],
                                         ['ab', 'TS(Tr)-256'],
                                         ['ab', 'TS(Ex)-256'],
                                         ['ab', 'TS(Ge)-256'],
                                         ['ib', 'rw', 'md', 'er', 'dc', 'TS(Re)-256'],
                                         ['ib', 'rw', 'md', 'er', 'dc', 'TS(Tr)-256'],
                                         ['ib', 'rw', 'md', 'er', 'dc', 'TS(Ex)-256'],
                                         ['ib', 'rw', 'md', 'er', 'dc', 'TS(Ge)-256'],
                                         ['ib(r-256)', 'rw(r-256)', 'md(r-256)', 'er(r-256)', 'dc(r-256)']],
                           "corpus-mh": [['w0', 'w1', 'w2', 'w3', 'w4'],
                                         ['s0', 's1', 's2', 's3', 's4'],
                                         ['ib', 'rw', 'md', 'er', 'dc']],
                           "corpus-ml": [['w0', 'w1', 'w2', 'w3', 'w4'],
                                         ['s0', 's1', 's2', 's3', 's4'],
                                         ['ib', 'rw', 'md', 'er', 'dc']]
                           }

    # Traversal according to different datasets
    for corpus in corpus_list:
        print("START:", '=' * 50, corpus, '=' * 50)
        for model_name in model_names:
            print("MODEL:", model_name)
            main(model_name, corpus, field_list_dict[corpus],
                     RI_field_lists_dict[corpus], en_topk=(10 if model_name == 'svm' else -1))
        print("END:", '=' * 50, corpus, '=' * 50)
