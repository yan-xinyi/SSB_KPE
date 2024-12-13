# -*- coding: utf-8 -*-

import pickle
import pandas as pd
from sklearn.model_selection import KFold
from configs import COLUMNS, fold_nums, sec_name_len, \
    sec_header_len, max_sen_nums, max_sen_len, sec_text_len
from utils import encoder_datas


def process_training_datas(input_text_type='for-bert-section'):
    """
    Process the training dataset for model training.

    :param input_text_type: Specifies the type of text input to process.
                            Default is 'for-bert-section', indicating input is suitable for BERT-based models.
    """

    # === Step 1: Load annotated data ===
    # Load the annotated dataset from an input file (tab-separated format)
    annotated_datas = pd.read_table('./inputs/input_datas.xls', names=COLUMNS)

    # === Step 2: Preprocess the data ===
    # Encode the dataset into the format required for training
    processed_datas, file_name = encoder_datas(
        datas=annotated_datas,
        sec_name_len=sec_name_len,
        sec_header_len=sec_header_len,
        sec_text_len=sec_text_len,
        max_sen_nums=max_sen_nums,
        max_sen_len=max_sen_len,
        input_text_type=input_text_type
    )

    # === Step 3: Split data into K folds ===
    # Initialize KFold for cross-validation with `fold_nums` splits
    kf = KFold(n_splits=fold_nums)

    # List to store training and testing data for each fold
    datas = []

    # Iterate through each fold, splitting data into training and testing sets
    for fold, (train_index, test_index) in enumerate(kf.split(processed_datas)):
        # Get training and testing data for the current fold
        train_datas = processed_datas.iloc[train_index]
        test_datas = processed_datas.iloc[test_index]

        # Extract dataset IDs for training and testing
        train_datasetids = processed_datas['dataset_id'][train_index]
        test_datasetids = processed_datas['dataset_id'][test_index]

        # Store fold data in a dictionary
        datas.append({
            "train_datas": train_datas,
            "test_datas": test_datas,
            "train_datasetids": train_datasetids,
            "test_datasetids": test_datasetids
        })

    # === Step 4: Save processed fold data ===
    # Save the fold data into a pickle file for later use
    pickle.dump(datas, open('./inputs/input_datas/fold_datas/%s.pkl' % file_name, 'wb'))


if __name__ == '__main__':
    # Execute the data processing function
    process_training_datas()
