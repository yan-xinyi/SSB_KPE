# -*- coding: utf-8 -*-
import pandas as pd
from tqdm import tqdm
from dataloader import data_loader
from utils import compute_metrics, encoder_datas
from model import *
from configs import *

# Prediction function
def predict(datatype='corpus-ml'):
    """
    Perform predictions on the specified dataset.

    :param datatype: The type of dataset to process and predict labels for.
                     Defaults to 'corpus-ml'.
    """
    # Load the annotated input data
    annotated_datas = pd.read_csv('./prediction_outputs/input_datas_%s.csv' % datatype, names=COLUMNS)

    # === Dataset Processing Parameters ===
    # Specify the type of input text (e.g., 'section')
    input_text_type = 'section'

    # Encode the dataset into a suitable format for the model
    processed_datas = encoder_datas(
        datas=annotated_datas,
        sec_name_len=sec_name_len,
        sec_header_len=sec_header_len,
        sec_text_len=sec_text_len,
        max_sen_nums=max_sen_nums,
        max_sen_len=max_sen_len,
        input_text_type=input_text_type,
        save_fold='./prediction_outputs'
    )

    # === Data Loading ===
    # Load the processed data into a DataLoader for batching
    dataloader = data_loader(processed_datas, batch_size=16)

    # Initialize a tensor to store prediction logits
    y_logits = torch.zeros((len(processed_datas), len(IDS2LABELS)))

    # Iterate through model folds for predictions
    for fold in range(0, 5):
        # === Model Definition ===
        # Define the TextCNN model
        model = TextCNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            device=DEVICE
        ).to(DEVICE)

        # Load pre-trained model weights
        model.load_state_dict(torch.load(
            open('./outputs/model/textcnn-32-128-2048-%s.bin' % fold, 'rb')
        )['model'])

        # === Model Testing ===
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for inference
            y_logits2, y_trues2, dataset_ids2 = [], [], []

            # Iterate through the DataLoader and process batches
            with tqdm(dataloader) as pbar_test:
                for inputs, labels in pbar_test:
                    outputs = model(inputs)  # Forward pass to get model outputs

                    # Compute performance metrics
                    p, r, f1 = compute_metrics(outputs, labels)
                    pbar_test.set_description("P:%s, R: %s, F1: %s" % (p, r, f1))

                    # Append results for evaluation
                    y_logits2.append(outputs.detach().cpu())
                    y_trues2.append(labels.detach().cpu())
                    dataset_ids2.append(inputs['dataset_ids'].detach().cpu())

        # === Evaluate Results ===
        y_logits2 = torch.cat(y_logits2, dim=0)
        y_trues2 = torch.cat(y_trues2, dim=0)
        dataset_ids2 = torch.cat(dataset_ids2, dim=0)

        # Compute metrics across datasets
        (pmc_p, pmc_r, pmc_f1), (lis_p, lis_r, lis_f1), \
        (ieee_p, ieee_r, ieee_f1), (all_p, all_r, all_f1) = compute_metrics(
            y_logits2, y_trues2, dataset_ids2, is_output_dataset_results=True
        )

        # Print overall performance metrics
        print(all_p, all_r, all_f1)

        # Accumulate logits from all folds
        y_logits += y_logits2

    # === Generate Predictions ===
    # Convert logits to predictions using softmax and label mapping
    preds = [IDS2LABELS[i] for i in torch.argmax(torch.softmax(y_logits, dim=-1), dim=-1).numpy().flatten()]

    # Update the labels in the original data with predictions
    for index, row in annotated_datas.iterrows():
        pred = preds[index]
        annotated_datas.loc[index, 'label'] = pred

    # Save the prediction results to a CSV file
    annotated_datas.to_csv("./prediction_outputs/output_datas_%s.csv" % datatype, index=False, header=False)
    print(annotated_datas.head(5))


# Main function to execute prediction
if __name__ == '__main__':
    predict()
