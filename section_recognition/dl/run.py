import os
import json
import time
import numpy as np
from configs import LOG_FOLD, fold_nums
from utils import set_seed

# === Step 1: Set random seed ===
# Ensures reproducibility of results by fixing the random seed
set_seed(2022)

# === Step 2: Define model names ===
# List of model architectures to be evaluated
model_names = ['textcnn', 'bilstm', 'bilstm-att']

# === Step 3: Iterate through models ===
# Loop through each model in the list for training and evaluation
for model_name in model_names:

    print("@" * 50, model_name, "@" * 50)  # Highlight the start of the model evaluation
    start_time = time.time()  # Record the start time for training and evaluation

    # Define input text type based on model name
    input_text_type = 'section'
    if model_name.startswith('hi'):
        input_text_type = 'sentence'

    # Containers to store evaluation results for different datasets and overall performance
    all_results = np.zeros((fold_nums, 3))  # Results for all datasets combined
    pmc_results = np.zeros((fold_nums, 3))  # Results for the PMC dataset
    lis_results = np.zeros((fold_nums, 3))  # Results for the LIS dataset
    ieee_results = np.zeros((fold_nums, 3))  # Results for the IEEE dataset

    # === Step 4: Iterate through folds ===
    for fold in range(fold_nums):

        print("*" * 40, fold + 1, "Start", "*" * 40)  # Indicate the start of fold processing

        # Train the model for the current fold
        # The `train.py` script is executed with specified parameters: model name, fold number, and input text type
        state = os.system("python train.py -m %s -f %s -i %s" % (model_name, fold, input_text_type))

        # === Step 5: Read performance metrics ===
        if state == 0:  # Check if the training script executed successfully
            results_path = os.path.join(LOG_FOLD, 'results.json')  # Path to results file
            results = json.load(open(results_path, 'r', encoding='utf-8'))  # Load evaluation metrics

            # Store results for each dataset and overall performance
            pmc_results[fold] = results[0]
            lis_results[fold] = results[1]
            ieee_results[fold] = results[2]
            all_results[fold] = results[3]

        # Print best results for the current fold
        print("Best_Results: P:%s, R: %s, F1: %s" % (all_results[fold][0], all_results[fold][1], all_results[fold][2]))
        print("*" * 40, fold + 1, "End", "*" * 40)

    # === Step 6: Compute and display average results ===
    # Calculate average precision, recall, and F1-score for each dataset and overall
    avg_pmc_results = np.average(pmc_results, axis=0)
    avg_lis_results = np.average(lis_results, axis=0)
    avg_ieee_results = np.average(ieee_results, axis=0)
    avg_all_results = np.average(all_results, axis=0)

    # Print detailed results for each dataset
    print("-" * 10, "PMC", "-" * 10)
    print(pmc_results)
    print("PMC:", avg_pmc_results)

    print("-" * 10, "LIS", "-" * 10)
    print(lis_results)
    print("LIS:", avg_lis_results)

    print("-" * 10, "IEEE", "-" * 10)
    print(ieee_results)
    print("IEEE:", avg_ieee_results)

    print("-" * 10, "ALL", "-" * 10)
    print(all_results)
    print("ALL:", avg_all_results)

    # === Step 7: Save results ===
    # Record training time and save results as a JSON file
    end_time = time.time()
    results = {
        "PMC": [pmc_results.tolist(), avg_pmc_results.tolist()],
        "LIS": [lis_results.tolist(), avg_lis_results.tolist()],
        "IEEE": [ieee_results.tolist(), avg_ieee_results.tolist()],
        "ALL": [all_results.tolist(), avg_all_results.tolist()],
        "training time": (end_time - start_time)  # Total time taken for training and evaluation
    }
    json.dump(results, open("./outputs/%s_results.json" % (model_name), 'w', encoding='utf-8'))  # Save results
