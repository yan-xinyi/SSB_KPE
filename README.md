# SSBKPE: Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information
This paper utilized the structural features and section texts obtained from the section structure information of academic articles to extract key phrases.

## Overview
<b>Data and source Code for the paper "SSBKPE: Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information".</b>

The rapid growth in academic publications has significantly increased the time required for researchers to identify relevant literature. Keyphrase Extraction (KPE) presents a promising solution by facilitating efficient access to critical academic resources. This research seeks to improve KPE models by using titles and abstracts as input data, yet there is limited exploration of full-text articles for this purpose. To fill this gap, the study utilizes structural features and section-specific texts from academic articles for keyphrase extraction. The initial phase involved examining the influence of seven structural features on KPE models. Following this, a keyphrase integration algorithm was used to merge extraction results from all sections, producing the final keyphrase set.

Moreover, the research assessed the impact of section structure identification quality on KPE performance. The findings reveal that integrating structural features enhances KPE, with distinct features affecting model performance differently. The keyphrase integration strategy delivered the most optimal results, and the quality of section structure identification also played a role in KPE effectiveness. These results underscore that exploiting section structure information from academic articles can significantly improve keyphrase extraction.


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
├─ Codes
│    ├─DL                                    # <b>Deep learning models</b>
│    │  ├──bertbilstmcrf.py                  # BERT-BiLSTM-CRF model implementation module
│    │  ├──bilstmcrf.py                      # BiLSTM-CRF model implementation module
│    │  ├──config.py                         # Config file
│    │  ├──obtain_results.py                 # Prediction results acquisition module
│    │  ├──preprocessing.py                  # Data preprocessing module
│    │  ├──split_dataset.py                  # Training and validation set segmentation module
│    │  ├──utils.py                          # library of auxiliary functions
│    │  ├──inputs                            # Folders for intermediate data
│    │  └──outputs                           # The folder where the output data is stored
│    └─ML                                    # <b>Traditional machine learning models</b>
│       ├──calculate_tf_tr_features.py       # tf*idf and textrank feature calculation module
│       ├──config.py                         # Config file
│       ├──crf.py                            # CRF Model Implementation Module
│       ├──crf_preprocessing.py              # CRF model data preprocessing module
│       ├──obtain_results.py                 # Prediction results acquisition module
│       ├──svm.py                            # SVM algorithm implementation module 
│       ├──svm_preprocessing.py              # SVM algorithm data preprocessing module
│       ├──textrank.py                       # Textrank algorithm implementation module
│       ├──tf_idf.py                         # Tf*idf algorithm implementation module
│       ├──training_glove.py                 # Glove word vector training module   
│       ├──utils.py                          # Library of auxiliary functions
│       ├──inputs                            # Folders for intermediate data
│       └──outputs                           # Folder where the output data is stored
└─ section_recognition
        ├─ML
        ├──config.py                         # Config file
        ├──model.ipynb                       # file of construction of models
        └──utils.py                          # Folders for intermediate data
        


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
In this paper, two classes of keyphrase extraction methods are selected to explore the role of chapter structure information on keyphrase extraction. One class is unsupervised keyphrase extraction methods based on TF*IDF and TextRank, and the other class is supervised key extraction methods based on Support Vector Machines, Conditional Random Fields, BiLSTM-CRF and BERT-BiLSTM-CRF.
### Implementation Steps for section structure

### Implementation Steps for SSB_KPE
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
    
>Chengzhi Zhang, Xinyi Yan, Lei Zhao, Yingyi Zhang. SSBKPE: Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information. ***Applied Soft Computing***, 2024 (Submitted).

