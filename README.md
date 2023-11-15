# Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information 
This paper utilized the structural features and section texts obtained from the section structure information of academic articles to extract key phrases.

## Overview
<b>Data and source Code for the paper "Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information".</b>

The exponential increase in academic papers has significantly increased the time required for researchers to access relevant literature. Keyphrase Extraction (KPE) offers a solution to this situation by enabling researchers to obtain required literature resources efficiently. The current study on KPE from academic articles aims to improve the performance of extraction models through innovative approaches using title and abstract as input corpora. However, research on input corpora is lacking, especially with regards to the use of full-text articles. To address this issue, this paper utilized the structural features and section texts obtained from the section structure information of academic articles to extract key phrases. The approach consists of two main parts:

  - exploring the effect of seven structural features on KPE models; 
  - integrating the extraction results from all section texts used as input corpora for KPE models via a keyphrase integration algorithm to obtain the keyphrase integration result.

Furthermore, this paper also examined the effect of the identification quality of section structure on the KPE performance. The results show that incorporating structural features is beneficial to KPE, but different structural features have varying effects on the performance of the models. The keyphrase integration result has the best performance, and the identification quality of section structure can affect KPE performance. These findings indicate that using the section structure information of academic articles contributes to effective KPE from academic articles.

## Directory Structure
<pre>SSB_AKE                                      # Root directory
├─ Dataset                                   # <b>Experimental datasets</b>
│   ├── IEEE-2000                            # dataset consists of 1316 articles from Pub-Med
│   │    ├── test
│   │    └── train
│   ├── LIS-2000                             # dataset consists of 2000 articles from Library and information science domain
│   │    ├── test           
│   │    └── train
│   └── PMC-1316                             # dataset consists of 2000 articles from Computer science domain
│        ├── test           
│        └── train
├─ CRF++                                     # a toolkit for conditional random fields (CRFs)
│    ├── README.MD                           # read this file to get into CRF++
│    ├── crf_learn.exe
│    ├── crf_text.exe
│    ├── exec.sh
│    └── libcrfpp.dll
├─dl                                         # <b>Deep learning models</b>
│  │  bertbilstmcrf.py                       # BERT-BiLSTM-CRF model implementation module
│  │  bilstmcrf.py                           # BiLSTM-CRF model implementation module
│  │  config.py                              # Config file
│  │  obtain_results.py                      # Prediction results acquisition module
│  │  preprocessing.py                       # Data preprocessing module
│  │  split_dataset.py                       # Training and validation set segmentation module
│  │  utils.py                               # library of auxiliary functions
│  ├─inputs                                  # Folders for intermediate data
│  └─outputs                                 # The folder where the output data is stored
├─ml                                         # <b>Traditional machine learning models</b>
│  │  calculate_tf_tr_features.py            # tf*idf and textrank feature calculation module
│  │  config.py                              # Config file
│  │  crf.py                                 # CRF Model Implementation Module
│  │  crf_preprocessing.py                   # CRF model data preprocessing module
│  │  obtain_results.py                      # Prediction results acquisition module
│  │  svm.py                                 # SVM algorithm implementation module 
│  │  svm_preprocessing.py                   # SVM algorithm data preprocessing module
│  │  textrank.py                            # Textrank algorithm implementation module
│  │  tf_idf.py                              # Tf*idf algorithm implementation module
│  │  training_glove.py                      # Glove word vector training module   
│  │  utils.py                               # Library of auxiliary functions
│  ├─inputs                                  # Folders for intermediate data
│  └─outputs                                 # Folder where the output data is stored
└─README.md
</pre>

## Dataset Discription
This paper utilized section structure information from academic articles to enhance KPE performance. Upon conducting a data investigation, it was observed that commonly used KPE datasets consist of academic articles presented as plain text, lacking differentiation between sections and paragraphs. To overcome this issue, there is a need to construct new datasets and segment the data based on the clear demarcation of sections within the articles.

<div align=center>
<b>Table 1. List of domains and journals/database of datasets</b>
  <img src="https://yan-xinyi.github.io/figures/SSB_KPE_1.png" width="750px" alt="Table 1. List of domains and journals/database of datasets">
</div>

Upon investigating the existing open-source datasets, it was observed that the HTML texts of each article within the PubMed dataset could be obtained directly from the PubMed website. In order to mitigate the issues of uniformity of section structures within a single domain, this study also selected academic articles from the fields of library and information science (LIS) and computer science (CS) as corpora for KPE. Following the completion of the data collection process, the academic articles with missing author's keyphrases are removed firstly. Subsequently, the HTML tags pertaining to paragraphs and headings within the articles were retained, while all other tags were removed. The details of the dataset are shown in Table 2. 

<div align=center>
<b>Table 2. Number of samples and author's keyphrases of training and test sets in different corpora.</b>
<img src="https://yan-xinyi.github.io/figures/SSB_KPE_2.png" width="750px" alt="Table 2. Number of samples and author's keyphrases of training and test sets in different corpora.">
</div>

## Citation
Please cite the following paper if you use this code and dataset in your work.
    
>Chengzhi Zhang, Xinyi Yan, Lei Zhao, Yingyi Zhang. Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information. ***Journal of the Association for Information Science and Technology***, 2023 (Submitting).

