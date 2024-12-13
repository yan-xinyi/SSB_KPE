# -*- coding: utf-8 -*-
import torch

# List of data column names for the dataset
COLUMNS = ['p_id', 'label', 'sec_title', 'sec_subtitle',
           'dataset_id', 'jname_id', 'bib_num', 'fn_num',
           'fig_num', 'tab_num', 'equ_num', 'para_num', 'sen_num',
           'word_num', 'sec_loc', 'sec_text']

# Mapping of section labels to numeric IDs for classification
LABELS2IDS = {
    "引言": 0,           # Introduction
    "相关工作": 1,        # Related Work
    "方法": 2,           # Methods
    "评估与结果": 3,      # Evaluation and Results
    "讨论与结论": 4,      # Discussion and Conclusion
    "其他": 5            # Others
}

# Reverse mapping of numeric IDs to section labels
IDS2LABELS = {val: key for key, val in LABELS2IDS.items()}

# Path to the pre-trained SciBERT model
BERT_PATH = '../scibert-model'

# Device configuration (use GPU if available, otherwise fallback to CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directory for storing output logs
LOG_FOLD = './outputs'

# === Parameter Settings ===

# Number of folds for cross-validation
fold_nums = 5

# === Dataset Processing Parameter Settings ===

# Maximum length for section names (e.g., "Introduction", "Methods")
sec_name_len = 32

# Maximum length for section headers (titles or subtitles)
sec_header_len = 128

# Maximum length for section text content
sec_text_len = 2048

# Maximum number of sentences per section
max_sen_nums = 80

# Maximum length of a single sentence
max_sen_len = 64

# === Model Parameter Settings ===

# Vocabulary size (number of unique tokens)
vocab_size = 31090

# Embedding dimension size (dimensionality of word embeddings)
embed_dim = 128

# Hidden dimension size (dimensionality of hidden layers)
hidden_dim = 128

# Number of output classes (corresponding to section labels)
num_classes = 6

# Dropout rate (probability of dropping neurons during training)
dropout = 0.5

# Sizes of convolutional kernels for CNN layers
kernel_sizes = [1, 2, 3, 4]

# === Training Parameter Settings ===

# Number of training epochs
epoches = 10

# Batch size for training
batch_size = 32

# Learning rate for the optimizer
lr = 1e-3

# Weight decay (L2 regularization term to prevent overfitting)
weight_decay = 1e-3

# Early stopping patience (number of epochs to wait before stopping if no improvement)
patience = 2
