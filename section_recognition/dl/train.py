# -*- coding: utf-8 -*-

import argparse
import json
import os.path
import pickle
import numpy as np
from torch import optim
from tqdm import tqdm
from configs import *
from dataloader import data_loader
from model import *
from utils import compute_metrics, set_seed

# Set the random seed for reproducibility
set_seed(2022)

# Initialize command-line arguments
def init_args():
    parser = argparse.ArgumentParser(description='Train a model')
    # Model parameters
    parser.add_argument('--model', '-m', help='Model type')
    parser.add_argument('--fold', '-f', help='Cross-validation fold')
    parser.add_argument('--input_text_type', '-i', help='Type of input data (e.g., section or sentence)')
    opt = parser.parse_args()
    return opt

# Training function
def train(model, optimizer, criterion, scheduler,
          train_dataloader, test_dataloader, epoches,
          input_text_type, model_name, fold,
          train_datasetids=None, test_datasetids=None, log_fold=LOG_FOLD):

    best_f1 = 0  # Variable to store the best F1-score
    results = np.zeros((4, 3))  # Container to store evaluation metrics (precision, recall, F1-score)

    # Loop through each epoch
    for epoch in range(epoches):
        model.train()  # Set the model to training mode
        losses = []  # Track training losses
        y_logits_train, y_trues_train = [], []  # Store model outputs and true labels for training data

        # Iterate over training batches
        with tqdm(train_dataloader) as pbar_train:
            for inputs, labels in pbar_train:
                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to prevent explosion
                optimizer.zero_grad()  # Reset gradients
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                pbar_train.set_description("Epoch(Train): %s, Loss: %s," % (epoch + 1, round(loss.item(), 2)))

                # Collect training metrics
                losses.append(loss.item())
                y_logits_train.append(outputs.detach().cpu())
                y_trues_train.append(labels.detach().cpu())

        # Calculate average loss and step scheduler
        loss = round(np.average(losses), 2)
        scheduler.step(np.average(losses))

        # Evaluate training metrics
        y_logits_train = torch.cat(y_logits_train, dim=0)
        y_trues_train = torch.cat(y_trues_train, dim=0)
        p, r, f1 = compute_metrics(y_logits_train, y_trues_train, is_output_dataset_results=False)
        print("Epoch(Train): %s, Loss: %s, P:%s, R: %s, F1: %s" % (epoch + 1, loss, p, r, f1))

        # Test the model
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            y_logits_test, y_trues_test = [], []  # Store model outputs and true labels for test data

            # Iterate over test batches
            with tqdm(test_dataloader) as pbar_test:
                for inputs, labels in pbar_test:
                    outputs = model(inputs)
                    pbar_test.set_description("Epoch(Test): %s" % (epoch + 1))
                    y_logits_test.append(outputs.detach().cpu())
                    y_trues_test.append(labels.detach().cpu())

            # Evaluate test metrics
            y_logits_test = torch.cat(y_logits_test, dim=0)
            y_trues_test = torch.cat(y_trues_test, dim=0)
            (pmc_p, pmc_r, pmc_f1), (lis_p, lis_r, lis_f1), \
            (ieee_p, ieee_r, ieee_f1), (all_p, all_r, all_f1) = compute_metrics(
                y_logits_test, y_trues_test, test_datasetids, is_output_dataset_results=True)
            print("Epoch(Test): %s, P:%s, R: %s, F1: %s" % (epoch + 1, all_p, all_r, all_f1))

            # Save the best model based on F1-score
            if all_f1 > best_f1:
                best_f1 = all_f1
                results[0] = [pmc_p, pmc_r, pmc_f1]
                results[1] = [lis_p, lis_r, lis_f1]
                results[2] = [ieee_p, ieee_r, ieee_f1]
                results[3] = [all_p, all_r, all_f1]

                # Save the model checkpoint
                if input_text_type == 'section':
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }, open("./outputs/model/%s-%s-%s-%s-%s.bin" % (
                        model_name, sec_name_len, sec_header_len, sec_text_len, fold), 'wb'))
                elif input_text_type == 'sentence':
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }, open("./outputs/model/%s-%s-%s-%s-%s-%s.bin" % (
                        model_name, sec_name_len, sec_header_len, max_sen_nums, max_sen_len, fold), 'wb'))

    # Save performance metrics
    results_path = os.path.join(log_fold, 'results.json')
    json.dump(results.tolist(), open(results_path, 'w', encoding='utf-8'))


if __name__ == '__main__':
    # Parse command-line arguments
    args = init_args()

    # Determine the data file path based on input text type
    file_name, file_path = '', ''
    save_fold = './inputs/input_datas/fold_datas'
    if args.input_text_type == 'section':
        file_name = "%s-%s-%s-%s.pkl" % (args.input_text_type, sec_name_len, sec_header_len, sec_text_len)
        file_path = os.path.join(save_fold, file_name)
    elif args.input_text_type == 'for-bert-section':
        sec_text_len = 512
        file_name = "%s-%s-%s-%s.pkl" % (args.input_text_type, sec_name_len, sec_header_len, sec_text_len)
        file_path = os.path.join(save_fold, file_name)

    # Load data
    datas = pickle.load(open(file_path, 'rb'))
    fold = int(args.fold)
    train_datas = datas[fold]['train_datas']
    test_datas = datas[fold]['test_datas']
    train_datasetids = datas[fold]['train_datasetids']
    test_datasetids = datas[fold]['test_datasetids']
    print("Training Data Shape:", train_datas.shape)
    print("Test Data Shape:", test_datas.shape)

    # Initialize data loaders
    train_dataloader = data_loader(train_datas, batch_size=batch_size)
    test_dataloader = data_loader(test_datas, batch_size=batch_size)

    # Initialize the model
    model = None
    if args.model == 'textcnn':
        model = TextCNN(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                        num_classes=num_classes, kernel_sizes=kernel_sizes, dropout=dropout, device=DEVICE).to(DEVICE)
    elif args.model == 'bilstm':
        model = BiLSTM(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                       num_classes=num_classes, kernel_sizes=kernel_sizes, dropout=dropout, device=DEVICE).to(DEVICE)
    elif args.model == 'bilstm-att':
        model = BiLSTM_Attention(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                                 num_classes=num_classes, kernel_sizes=kernel_sizes, dropout=dropout, device=DEVICE).to(DEVICE)

    print(model)

    # Define optimizer, loss function, and learning rate scheduler
    no_decay = ['bias', 'gamma', 'beta']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=True)

    # Train the model
    train(model=model,
          optimizer=optimizer,
          criterion=criterion,
          scheduler=scheduler,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          epoches=epoches,
          model_name=args.model,
          fold=args.fold,
          train_datasetids=train_datasetids,
          test_datasetids=test_datasetids,
          input_text_type=args.input_text_type)
