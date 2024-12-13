# -*- coding: utf-8 -*-

import json
import numpy as np
import torch
from nltk import sent_tokenize, word_tokenize
from torch.utils.data import Dataset, DataLoader
from configs import BERT_PATH, LABELS2IDS, DEVICE  # Import configuration constants
from transformers import BertTokenizer


class TextDataSet(Dataset):
    """
    Custom PyTorch Dataset for handling text data with associated metadata and labels.
    This class is designed to process structured data for use in neural networks.
    """

    def __init__(self, datas):
        """
        Initialize the dataset with input data.

        Args:
            datas (DataFrame): A Pandas DataFrame containing the data.
                Expected columns include:
                - p_id, label, sec_title_id, sec_subtitle_id, sec_text_id
                - Various numerical features like bib_num, fn_num, etc.
                - Metadata fields like dataset_id, jname_id, etc.
        """
        super(TextDataSet, self).__init__()

        # Assigning the data to the dataset
        self.datas = datas

        # Primary IDs
        self.p_ids = self.datas['p_id'].values
        self.labels = self.datas['label'].values  # Labels for the dataset (target values)

        # Textual feature IDs (stored as JSON strings)
        self.sec_title_ids = self.datas['sec_title_id'].values
        self.sec_subtitle_ids = self.datas['sec_subtitle_id'].values
        self.sec_text_ids = self.datas['sec_text_id'].values

        # Metadata fields (reshaped into list format for tensor conversion)
        self.dataset_ids = self.datas['dataset_id'].values.reshape(-1, 1).tolist()
        self.jname_ids = self.datas['jname_id'].values.reshape(-1, 1).tolist()

        # Various numeric features (e.g., counts of bibliography, figures, etc.)
        self.bib_nums = self.datas['bib_num'].values.reshape(-1, 1).tolist()
        self.fn_nums = self.datas['fn_num'].values.reshape(-1, 1).tolist()
        self.fig_nums = self.datas['fig_num'].values.reshape(-1, 1).tolist()
        self.tab_nums = self.datas['tab_num'].values.reshape(-1, 1).tolist()
        self.equ_nums = self.datas['equ_num'].values.reshape(-1, 1).tolist()

        # Document structure features
        self.para_nums = self.datas['para_num'].values.reshape(-1, 1).tolist()
        self.sen_nums = self.datas['sen_num'].values.reshape(-1, 1).tolist()
        self.word_nums = self.datas['word_num'].values.reshape(-1, 1).tolist()

        # Section location within the document
        self.sec_locs = self.datas['sec_loc'].values.reshape(-1, 1).tolist()

    def __getitem__(self, item):
        """
        Retrieve a single data point and its corresponding label.

        Args:
            item (int): Index of the data point to retrieve.

        Returns:
            Tuple:
                - Dictionary containing the processed features as tensors.
                - Tensor containing the corresponding label.
        """
        # Target label for the current item
        labels = self.labels[item]

        # Load JSON strings and convert to Python objects (lists)
        sec_title_id = json.loads(self.sec_title_ids[item])
        sec_subtitle_id = json.loads(self.sec_subtitle_ids[item])
        sec_text_id = json.loads(self.sec_text_ids[item])

        # Return a dictionary of features and the corresponding label
        return {
            "sec_title_ids": torch.tensor(sec_title_id).long().to(DEVICE),  # Section title tokens
            "sec_subtitle_ids": torch.tensor(sec_subtitle_id).long().to(DEVICE),  # Section subtitle tokens
            "sec_text_ids": torch.tensor(sec_text_id).long().to(DEVICE),  # Section text tokens

            # Metadata features
            "dataset_ids": torch.tensor(self.dataset_ids[item]).float().to(DEVICE),
            "jname_ids": torch.tensor(self.jname_ids[item]).float().to(DEVICE),

            # Numerical features
            "bib_nums": torch.tensor(self.bib_nums[item]).float().to(DEVICE),
            "fn_nums": torch.tensor(self.fn_nums[item]).float().to(DEVICE),
            "fig_nums": torch.tensor(self.fig_nums[item]).float().to(DEVICE),
            "tab_nums": torch.tensor(self.tab_nums[item]).float().to(DEVICE),
            "equ_nums": torch.tensor(self.equ_nums[item]).float().to(DEVICE),

            # Structural features
            "para_nums": torch.tensor(self.para_nums[item]).float().to(DEVICE),
            "sen_nums": torch.tensor(self.sen_nums[item]).float().to(DEVICE),
            "word_nums": torch.tensor(self.word_nums[item]).float().to(DEVICE),

            # Section location feature
            "sec_locs": torch.tensor(self.sec_locs[item]).float().to(DEVICE),
        }, torch.tensor(labels).long().to(DEVICE)

    def __len__(self):
        """
        Return the total number of data points in the dataset.

        Returns:
            int: Number of data points in the dataset.
        """
        return self.datas.shape[0]


# Function to load data using PyTorch DataLoader
def data_loader(datas, batch_size=128, shuffle=False):
    """
    Create a DataLoader for the dataset.

    Args:
        datas (DataFrame): A Pandas DataFrame containing the data.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at the beginning of each epoch.

    Returns:
        DataLoader: PyTorch DataLoader for iterating over the dataset.
    """
    dataset = TextDataSet(datas)  # Create the dataset
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)  # Initialize DataLoader
    return dataloader
