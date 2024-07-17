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
│    ├── IEEE-2000                           # dataset consists of 1316 articles from Pub-Med
│    │    ├── test
│    │    └── train
│    ├── LIS-2000                            # dataset consists of 2000 articles from Library and information science domain
│    │    ├── test           
│    │    └── train
│    └── PMC-1316                            # dataset consists of 2000 articles from Computer science domain
│        ├── test           
│        └── train
├─ CRF++                                     # a toolkit for conditional random fields (CRFs)
│    ├── README.MD                           # read this file to get into CRF++
│    ├── crf_learn.exe
│    ├── crf_text.exe
│    ├── exec.sh
│    └── libcrfpp.dll
└─ Codes
     ├─DL                                    # <b>Deep learning models</b>
     │  │  bertbilstmcrf.py                  # BERT-BiLSTM-CRF model implementation module
     │  │  bilstmcrf.py                      # BiLSTM-CRF model implementation module
     │  │  config.py                         # Config file
     │  │  obtain_results.py                 # Prediction results acquisition module
     │  │  preprocessing.py                  # Data preprocessing module
     │  │  split_dataset.py                  # Training and validation set segmentation module
     │  │  utils.py                          # library of auxiliary functions
     │  ├─inputs                             # Folders for intermediate data
     │  └─outputs                            # The folder where the output data is stored
     └─ML                                    # <b>Traditional machine learning models</b>
        │  calculate_tf_tr_features.py       # tf*idf and textrank feature calculation module
        │  config.py                         # Config file
        │  crf.py                            # CRF Model Implementation Module
        │  crf_preprocessing.py              # CRF model data preprocessing module
        │  obtain_results.py                 # Prediction results acquisition module
        │  svm.py                            # SVM algorithm implementation module 
        │  svm_preprocessing.py              # SVM algorithm data preprocessing module
        │  textrank.py                       # Textrank algorithm implementation module
        │  tf_idf.py                         # Tf*idf algorithm implementation module
        │  training_glove.py                 # Glove word vector training module   
        │  utils.py                          # Library of auxiliary functions
        ├─inputs                             # Folders for intermediate data
        └─outputs                            # Folder where the output data is stored


</pre>

## Dataset Discription
This study leveraged the section structure information of academic articles to enhance Keyphrase Extraction (KPE) performance. To address the issue of uniform section structures within a single domain, we selected academic articles from both Library and Information Science (LIS) and Computer Science (CS) for our KPE corpus. Given the substantial volume of full-text academic papers, we opted to randomly shuffle the data and split it into training, validation, and test sets in an 8:1:1 ratio, rather than employing ten-fold cross-validation. The number of samples and author-provided keyphrases for each set are detailed in Table 1.

<div align=center>
<b>Table 1. Number of samples and author's keyphrases of training, valid and test sets in different corpora.</b>
<img src="https://github.com/yan-xinyi/images/blob/main/SSB_KPE/SSB_KPE_1.png" width="75%" alt="Table 1. Number of samples and author's keyphrases of training and test sets in different corpora."><br>
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
    
>Chengzhi Zhang, Xinyi Yan, Lei Zhao, Yingyi Zhang. Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information. ***Journal of the Association for Information Science and Technology***, 2023 (Submitted).

