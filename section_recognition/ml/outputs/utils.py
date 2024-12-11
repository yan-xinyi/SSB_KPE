# -*- coding: utf-8 -*-
# @Time    : 2022/4/15 14:30
# @Author  : leizhao150
import os
import pickle
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, chi2, VarianceThreshold
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MaxAbsScaler
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# Define stopwords and punctuation for text processing
stop_words = stopwords.words('english')
english_punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')', '[', ']',
                        '&', '!', '*', '@', '#', '$', '%','\\','\"','}','{', '-',
                        '–', '—', '..', '/', '=', '∣', '…', '′', '⋅', '×', '+', '•']
porterStemmer = PorterStemmer()
spacy_model = spacy.load('en_core_web_sm')


# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# Function to calculate precision, recall, and F1 score
def metrics(y_true, y_pred):
    p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # print(classification_report(y_true, y_pred, zero_division=0))
    return round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)


# Function to execute the model using cross-validation
def execute_model(model, xs, ys, dataset_ids = None, info="SVM", fold_nums=5, is_output_dataset_results=False):
    # Perform K-Fold cross-validation (5 splits)
    kf = KFold(n_splits=5)  # 分成5份
    if is_output_dataset_results == False:
        results = np.zeros((fold_nums, 3))
        all_preds, all_targets = [], []
        for fold, (train_index, test_index) in enumerate(kf.split(xs)):
            # Split the data into training and testing sets for each fold
            x_train, y_train = xs[train_index], ys[train_index]
            x_test, y_test = xs[test_index], ys[test_index]

            # Normalize the data using MaxAbsScaler
            scaler = MaxAbsScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Train the model and predict on the test set
            clf = OneVsRestClassifier(model)             # OneVsRest strategy for multiclass classification
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            # Calculate performance metrics for this fold
            p, r, f1 = metrics(y_test, y_pred)

            results[fold] = [p, r, f1]

            all_preds.extend(y_pred)
            all_targets.extend(y_test)

        # Compute the average results across all folds
        avg_results = np.round(np.average(results, axis=0), 2)
        print("-" * 50)
        print(results)
        print("INFO: %s, AVG->P:%s, R:%s, F1:%s" % (info, avg_results[0], avg_results[1], avg_results[2]))
        print(classification_report(all_targets, all_preds, zero_division=0, digits=4))
        print("=" * 50)
        return avg_results[0], avg_results[1], avg_results[2]
    else:
        # If output dataset results are required, initialize arrays to store results for different datasets
        all_results = np.zeros((fold_nums, 3))
        pmc_results = np.zeros((fold_nums, 3))
        lis_results = np.zeros((fold_nums, 3))
        ieee_results = np.zeros((fold_nums, 3))
        all_preds, all_targets = [], []
        reports = []
        for fold, (train_index, test_index) in enumerate(kf.split(xs)):
            x_train, y_train = xs[train_index], ys[train_index]
            x_test, y_test = xs[test_index], ys[test_index]
            dataset_ids_train, dataset_ids_test = dataset_ids[train_index], dataset_ids[test_index]

            # Normalize the data using MaxAbsScaler
            scaler = MaxAbsScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # Train the model and predict on the test set
            clf = OneVsRestClassifier(model)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            # Collect results for different datasets
            # (additional code for dataset-specific result collection follows)
            pmc_y_true, pmc_y_pred = [], []
            lis_y_true, lis_y_pred = [], []
            ieee_y_true, ieee_y_pred = [], []
            for _label, _pred, _dataset_id in zip(y_test, y_pred, dataset_ids_test):
                if _dataset_id == 1:
                    pmc_y_true.append(_label)
                    pmc_y_pred.append(_pred)
                elif _dataset_id == 2:
                    lis_y_true.append(_label)
                    lis_y_pred.append(_pred)
                elif _dataset_id == 3:
                    ieee_y_true.append(_label)
                    ieee_y_pred.append(_pred)

            pmc_p, pmc_r, pmc_f1 = metrics(pmc_y_true, pmc_y_pred)
            lis_p, lis_r, lis_f1 = metrics(lis_y_true, lis_y_pred)
            ieee_p, ieee_r, ieee_f1 = metrics(ieee_y_true, ieee_y_pred)
            all_p, all_r, all_f1 = metrics(y_test, y_pred)

            pmc_results[fold] = [pmc_p, pmc_r, pmc_f1]
            lis_results[fold] = [lis_p, lis_r, lis_f1]
            ieee_results[fold] = [ieee_p, ieee_r, ieee_f1]
            all_results[fold] = [all_p, all_r, all_f1]

            all_preds.extend(y_pred)
            all_targets.extend(y_test)
            # print('-'*20, fold, '-'*20)
            rr = classification_report(y_test, y_pred, zero_division=0, digits=4, output_dict=True)
            report = []
            for item in ['0', '1', '2', '3', '4', '5']:
                precision = rr[item]['precision']
                recall = rr[item]['recall']
                f1_score = rr[item]['f1-score']
                report.append([precision, recall, f1_score])
            reports.append(report)

        avg_pmc_results = np.round(np.average(pmc_results, axis=0), 2)
        avg_lis_results = np.round(np.average(lis_results, axis=0), 2)
        avg_ieee_results = np.round(np.average(ieee_results, axis=0), 2)
        avg_all_results = np.round(np.average(all_results, axis=0), 2)

        print("-" * 50)
        print(pmc_results)
        print("INFO: %s, PMC->P:%s, R:%s, F1:%s" % (info, avg_pmc_results[0], avg_pmc_results[1], avg_pmc_results[2]))
        print("-" * 40)

        print(lis_results)
        print("INFO: %s, LIS->P:%s, R:%s, F1:%s" % (info, avg_lis_results[0], avg_lis_results[1], avg_lis_results[2]))
        print("-" * 40)

        print(ieee_results)
        print("INFO: %s, IEEE->P:%s, R:%s, F1:%s" % (info, avg_ieee_results[0], avg_ieee_results[1], avg_ieee_results[2]))
        print("-" * 40)

        print(all_results)
        print("INFO: %s, ALL->P:%s, R:%s, F1:%s" % (info, avg_all_results[0], avg_all_results[1], avg_all_results[2]))
        print("-" * 40)

        reports = np.array(reports)
        print(np.mean(reports, axis=0)*100)
        print(classification_report(all_targets, all_preds, zero_division=0, digits=4))
        print("=" * 50)
        return avg_all_results[0], avg_all_results[1], avg_all_results[2]


# Preprocess titles
def preprocessing_titles(texts, ys=None, feature_selection_approach='CHI',
                         percentile=10, save_name='sec_title_words',
                         rv_fea_num=False):
    processed_datas = []
    file_path = "./outputs/%s.pkl" % save_name
    if not os.path.exists(file_path):
        for text in tqdm(texts):
            text = text.lower().strip()
            # Tokenization
            seg_words = [token.text.strip() for token in spacy_model(text) if len(token.text.strip()) > 0]
            # Remove stop words and punctuation
            seg_words = [word for word in seg_words if (word not in stop_words) and (word not in english_punctuations)]
            # Remove numerical identifiers in titles
            seg_words = [word for word in seg_words if not re.search("[\d\.]+", word)]
            # Apply stemming
            seg_words = [porterStemmer.stem(word) for word in seg_words]
            processed_datas.append(" ".join(seg_words))
        # Save processed results
        pickle.dump(processed_datas, open(file_path, 'wb'))
    else:
        processed_datas = pickle.load(open(file_path, 'rb'))

    # Vectorization - TF*IDF
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    xs = vectorizer.fit_transform(processed_datas)
    if rv_fea_num: return xs.shape[1]

    if str(ys) != 'None' and ys.all() != None:
        # Feature selection
        if feature_selection_approach == "CHI":
            # Select top features based on Chi-Square test
            xs = SelectPercentile(chi2, percentile=percentile).fit_transform(xs, ys)
        elif feature_selection_approach == "IG":
            # Select top features based on mutual information
            xs = SelectPercentile(mutual_info_classif, percentile=percentile).fit_transform(xs, ys)
        else:
            # Remove features with low variance
            xs = VarianceThreshold(threshold=0).fit_transform(xs)
    return xs


# Preprocess text
df_count = {key: val for key, val in pd.read_csv('./outputs/df.csv', names=['word', 'freq']).values}
def preprocessing_text(texts, ys=None, feature_selection_approach='CHI',
                       percentile=10, save_name='sec_text_words', rv_fea_num=False):
    processed_datas = []
    file_path = "./outputs/%s.pkl" % save_name
    if not os.path.exists(file_path):
        for text in tqdm(texts):
            text = text.strip().lower()
            # Filter out URLs
            text = re.sub('^(http|https|ftp)\://([a-zA-Z0-9\.\-]+(\:[a-zA-Z0-9\.&%\$\-]+)*@)?((25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9])\.(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9]|0)\.(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[1-9]|0)\.(25[0-5]|2[0-4][0-9]|[0-1]{1}[0-9]{2}|[1-9]{1}[0-9]{1}|[0-9])|([a-zA-Z0-9\-]+\.)*[a-zA-Z0-9\-]+\.[a-zA-Z]{2,4})(\:[0-9]+)?(/[^/][a-zA-Z0-9\.\,\?\'\\/\+&%\$#\=~_\-@]*)*$', '', text)
            text = re.sub('^([a-zA-Z]\:|\\\\[^\/\\:*?"<>|]+\\[^\/\\:*?"<>|]+)(\\[^\/\\:*?"<>|]+)+(\.[^\/\\:*?"<>|]+)$', '', text)
            # Filter out citation tags
            text = re.sub("[\[\]\d\-,\.]+", "", text)
            # Tokenization
            seg_words = [token.text.strip() for token in spacy_model(text) if len(token.text.strip()) > 0]
            # Remove stop words
            seg_words = [word for word in seg_words if (word not in stop_words) and (word not in english_punctuations)]
            # Apply stemming
            seg_words = [porterStemmer.stem(word) for word in seg_words]
            # Remove overly long words
            seg_words = [word for word in seg_words if len(word) <= 32]
            # Remove low-frequency and high-document-frequency words
            seg_words = [word for word, val in dict(Counter(seg_words)).items() if val > 3]
            seg_words = [word for word in seg_words if word in df_count.keys() and df_count[word] < df_count['N'] / 2]
            processed_datas.append(" ".join(seg_words))
        # Save processed results
        pickle.dump(processed_datas, open(file_path, 'wb'))
    else:
        processed_datas = pickle.load(open(file_path, 'rb'))

    # Vectorization - TF*IDF
    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    xs = vectorizer.fit_transform(processed_datas)
    if rv_fea_num: return xs.shape[1]

    # Feature selection
    if feature_selection_approach == "CHI":
        # Select top features based on Chi-Square test
        xs = SelectPercentile(chi2, percentile=percentile).fit_transform(xs, ys)
    elif feature_selection_approach == "IG":
        # Select top features based on mutual information
        xs = SelectPercentile(mutual_info_classif, percentile=percentile).fit_transform(xs, ys)
    else:
        # Remove features with low variance
        xs = VarianceThreshold(threshold=0).fit_transform(xs)
    return xs