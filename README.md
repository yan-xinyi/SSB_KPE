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
└─scibert-model                              # Folder where the scibert model is stored

</pre>

## Dataset Discription
This paper utilized section structure information from academic articles to enhance KPE performance. Upon conducting a data investigation, it was observed that commonly used KPE datasets consist of academic articles presented as plain text, lacking differentiation between sections and paragraphs. To overcome this issue, there is a need to construct new datasets and segment the data based on the clear demarcation of sections within the articles.

<div align=left>
<b>Table 1. List of domains and journals/database of datasets</b><br>
  <img src="https://yan-xinyi.github.io/figures/SSB_KPE_1.png" width="75%" alt="Table 1. List of domains and journals/database of datasets"><br>
  <b>Note.</b> 1: https://www.ncbi.nlm.nih.gov/pmc/<br><br>
</div>



Upon investigating the existing open-source datasets, it was observed that the HTML texts of each article within the PubMed dataset could be obtained directly from the PubMed website. In order to mitigate the issues of uniformity of section structures within a single domain, this study also selected academic articles from the fields of library and information science (LIS) and computer science (CS) as corpora for KPE. Following the completion of the data collection process, the academic articles with missing author's keyphrases are removed firstly. Subsequently, the HTML tags pertaining to paragraphs and headings within the articles were retained, while all other tags were removed. The details of the dataset are shown in Table 2. 

<div align=left>
<b>Table 2. Number of samples and author's keyphrases of training and test sets in different corpora.</b>
<img src="https://yan-xinyi.github.io/figures/SSB_KPE_2.png" width="75%" alt="Table 2. Number of samples and author's keyphrases of training and test sets in different corpora."><br>
</div>


## Requirements
System environment is set up according to the following configuration:
- Python==3.7
- Torch==1.8.0
- torchvision==0.9.0
- Sklearn==0.0
- Numpy 1.25.1+mkl
- nltk==3.6.2
- Tqdm==4.56.0

## Quick Start
In this paper, two classes of keyword extraction methods are selected to explore the role of chapter structure information on keyword extraction. One class is unsupervised keyword extraction methods based on TF*IDF and TextRank, and the other class is supervised key extraction methods based on Support Vector Machines, Conditional Random Fields, BiLSTM-CRF and BERT-BiLSTM-CRF.
### Implementation Steps for machine learing model
1. <b>Processing:</b> Run the processing.py file to process the data into json format:
    `python processing.py`

   The data is preprocessed to the format like: {['word','Value_et1',... ,'Value_et17','Value_eeg1',... ,'Value_eeg8','tag']}

2. <b>Configuration:</b> Configure hyperparameters in the `config.py` file. There are roughly the following parameters to set:
    - `modeltype`: select which model to use for training and testing.
    - `train_path`,`test_path`,`vocab_path`,`save_path`: path of train data, test data, vocab data and results.
    - `fs_name`, `fs_num`: Name and number of cognitive traits.
    - `run_times`: Number of repetitions of training and testing.
    - `epochs`: refers to the number of times the entire training dataset is passed through the model during the training process. 
    - `lr`: learning rate.
    - `vocab_size`: the size of vocabulary. 37347 for Election-Trec Dataset, 85535 for General-Twitter.
    - `embed_dim`,`hidden_dim`: dim of embedding layer and hidden layer.
    - `batch_size`: refers to the number of examples (or samples) that are processed together in a single forward/backward pass during the training or inference process of a machine learning model.
    - `max_length`: is a parameter that specifies the maximum length (number of tokens) allowed for a sequence of text input. It is often used in natural language processing tasks, such as text generation or text classification.
3. <b>Modeling:</b> Modifying combinations of additive cognitive features in the model.

   For example, the code below means add all 25 features into the model:

         `input = torch.cat([input, inputs['et'], inputs['eeg']], dim=-1)`
5. <b>Training and testing:</b> based on your system, open the terminal in the root directory 'AKE' and type this command:
    `python main.py` 



## Citation
Please cite the following paper if you use this code and dataset in your work.
    
>Chengzhi Zhang, Xinyi Yan, Lei Zhao, Yingyi Zhang. Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information. ***Journal of the Association for Information Science and Technology***, 2023 (Submitting).

