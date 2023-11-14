# Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information 
This paper utilized the structural features and section texts obtained from the section structure information of academic articles to extract key phrases.

## Overview
<b>Data and source Code for the paper "Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information".</b>

The exponential increase in academic papers has significantly increased the time required for researchers to access relevant literature. Keyphrase Extraction (KPE) offers a solution to this situation by enabling researchers to obtain required literature resources efficiently. The current study on KPE from academic articles aims to improve the performance of extraction models through innovative approaches using title and abstract as input corpora. However, research on input corpora is lacking, especially with regards to the use of full-text articles. To address this issue, this paper utilized the structural features and section texts obtained from the section structure information of academic articles to extract key phrases. The approach consists of two main parts:

  - exploring the effect of seven structural features on KPE models; 
  - integrating the extraction results from all section texts used as input corpora for KPE models via a keyphrase integration algorithm to obtain the keyphrase integration result.

Furthermore, this paper also examined the effect of the identification quality of section structure on the KPE performance. The results show that incorporating structural features is beneficial to KPE, but different structural features have varying effects on the performance of the models. The keyphrase integration result has the best performance, and the identification quality of section structure can affect KPE performance. These findings indicate that using the section structure information of academic articles contributes to effective KPE from academic articles.

## Directory Structure
<pre>SSB_AKE                                          Root directory
├── Dataset                                  Experimental datasets
│   ├── IEEE-2000                            dataset consists of 1316 articles from Pub-Med
│   │    ├── test
│   │    └── train
│   ├── LIS-2000                             dataset consists of 2000 articles from Library and information science domain
│   │    ├── test           
│   │    └── train
│   └── PMC-1316                             dataset consists of 2000 articles from Computer science domain
│        ├── test           
│        └── train
├── ML                                       Module of the machine learning models
│   ├── CRF++                                a toolkit for conditional random fields (CRFs)
│   │    ├── README.MD                       read this file to get into CRF++
│   │    ├── crf_learn.exe
│   │    ├── crf_text.exe
│   │    ├── exec.sh
│   │    └── libcrfpp.dll
│   ├── preprocessing.py                     Segmentation, word tagging, deactivation, etc
│   ├── config.py                            Get the absolute path under the current working directory
│   ├── utils.py                             Read and save the data, merge phrases into a list
│   ├── biuld_path.py                        Create path for reasing, saving datas
│   ├── tf-idf.py                            Constructing a TF-IDF based KPE model
│   ├── textrank.py                          Constructing a Textrank based KPE model
│   ├── naivebayes.py                        Constructing a naive bayes based KPE model
│   ├── bilstm+crf.py                        Constructing a Bilstm+crf based KPE model
│   ├── crf.py                               Constructing a crf based KPE model
│   └── evaluate.py                          Calculate the P, R and F1 values of the extraction datas
├── DL                                       Module of the machine learning models
│   ├── CRF++                                a toolkit for conditional random fields (CRFs)
│   │    ├── README.MD                       read this file to get into CRF++
│   │    ├── crf_learn.exe
│   │    ├── crf_text.exe
│   │    ├── exec.sh
│   │    └── libcrfpp.dll
│   ├── preprocessing.py                     Segmentation, word tagging, deactivation, etc
│   ├── config.py                            Get the absolute path under the current working directory
│   ├── utils.py                             Read and save the data, merge phrases into a list
│   ├── biuld_path.py                        Create path for reasing, saving datas
│   ├── tf-idf.py                            Constructing a TF-IDF based KPE model
│   ├── textrank.py                          Constructing a Textrank based KPE model
│   ├── naivebayes.py                        Constructing a naive bayes based KPE model
│   ├── bilstm+crf.py                        Constructing a Bilstm+crf based KPE model
│   ├── crf.py                               Constructing a crf based KPE model
│   └── evaluate.py                          Calculate the P, R and F1 values of the extraction datas
├── config.py                                Path configuration file
├── utils.py                                 Some auxiliary functions
├── evaluate.py                              Surce code for result evaluation
├── processing.py                            Source code of preprocessing function
├── main.py                                  Surce code for main function
└─README.md
</pre>

## Dataset Discription
In our study, two kinds of data are used: the cognitive signal data from human readings behaviors and the AKE from Microblogs data.
### 1. Cognitive Signal Data -- ZUCO Dataset
In this study, we choose <b>the Zurich Cognitive Language Processing Corpus ([ZUCO](https://www.nature.com/articles/sdata2018291))</b>, which captures eye-tracking signals and EEG signals of 12 adult native speakers reading approximately 1100 English sentences in normal and task reading modes. The raw data can be visited at: https://osf.io/2urht/#!. 

Only data from <b>the normal reading mode</b> were utilized to align with human natural reading habits. The reading corpus includes two datasets: 400 movie reviews from the Stanford Sentiment Treebank and 300 paragraphs about celebrities from the Wikipedia Relation Extraction Corpus. We release our all train and test data in “dataset” directory, In the ZUCO dataset, cognitive features have been spliced between each word and the corresponding label. 

Specifically, there are <b>17 Eye-tracking features</b> and <b>8 EEG features</b> were extracted from the dataset:

- <b>Eye-tracking features</b>
  In ZUCO Corpus, Hollenstein et al.(2019) categorized the 17 eye-tracking features into three groups(Refer to Table 1): Early-Stage Features,Late-Stage Features and Contextual Features, encompassing all gaze behavior stages and contextual influences.
    - Early-Stage Features reflect readers' initial comprehension and cognitive processing of the text.
    - Late-Stage Features indicate readers' syntactic and semantic comprehension.
    - Contextual Features refer to the gaze behavior of readers on the words surrounding the current word.


<div align=center>
Table 1. Summary of Eye-Tracking Features
<img src="https://yan-xinyi.github.io/figures/ET_features.png" width="750px" alt="Table 1. Summary of Eye-Tracking Features">
</div>


## Citation
Please cite the following paper if you use this code and dataset in your work.
    
>Chengzhi Zhang, Xinyi Yan, Lei Zhao, Yingyi Zhang. Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information. ***Journal of the Association for Information Science and Technology***, 2023 (Submitting).

