# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertConfig

from configs import BERT_PATH
from utils import set_seed

# set random seed
set_seed(2022)

# CNN encoding
class CNNEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, kernel_sizes=None, dropout=0.1):
        super(CNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.kernel_sizes = [1, 2, 3, 4] if kernel_sizes == None else kernel_sizes

        self.cnns = nn.ModuleList([nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=(kernel_size,)),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1),
        ) for kernel_size in self.kernel_sizes])

        self.layernorm = nn.LayerNorm(
            normalized_shape=(self.hidden_dim * len(self.kernel_sizes)))
        self.output = nn.Linear(self.hidden_dim * len(self.kernel_sizes), self.output_dim)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        temp = []
        inputs = torch.transpose(inputs, 1, 2) #[B, E, S]
        for cnn in self.cnns:
            cnn_result = cnn(inputs).squeeze(-1)
            temp.append(cnn_result)
        cnn_result = self.dropout(
            F.relu(self.layernorm(torch.cat(temp, dim=-1))))
        outputs = self.output(cnn_result)
        return outputs

class Non_semantic_feature_fusion(nn.Module):
    """
    A PyTorch module for fusing non-semantic features with text embeddings.
    This class integrates additional non-semantic features (e.g., metadata or statistics)
    with the output of a text representation model to enhance downstream tasks.

    Args:
        input_dim (int): The dimension of the input features (e.g., text embeddings).
        output_dim (int): The desired output dimension after feature fusion.
        features (list, optional): A list of feature names to be fused. Defaults to a
            predefined list of non-semantic feature names.
        dropout (float, optional): The dropout rate applied during training. Default is 0.1.
    """
    def __init__(self, input_dim, output_dim, features=None, dropout=0.1):
        super(Non_semantic_feature_fusion, self).__init__()

        # Initialize the input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize the list of features, or use a default set of features if not provided
        self.features = features
        self.dropout_rate = dropout
        if self.features is None:
            # Default set of non-semantic feature names
            self.features = ['dataset_ids', 'jname_ids', 'bib_nums', 'fn_nums',
                             'fig_nums', 'tab_nums', 'equ_nums', 'para_nums',
                             'sen_nums', 'word_nums', 'sec_locs']

        # Layer normalization for input normalization
        self.layernorm = nn.LayerNorm(input_dim + len(self.features))

        # Fully connected layer for transforming fused features to the desired output dimension
        self.linear = nn.Linear(input_dim + len(self.features), output_dim)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, last_state, inputs):
        """
        Forward pass of the Non_semantic_feature_fusion module.

        Args:
            last_state (torch.Tensor): The final hidden state from the text representation model
                (e.g., BERT, RoBERTa), with shape (batch_size, input_dim).
            inputs (dict): A dictionary containing the additional non-semantic features
                to be fused, where each feature is a tensor with shape (batch_size, 1).

        Returns:
            torch.Tensor: The output tensor after fusing the features and applying the
            linear transformation, with shape (batch_size, output_dim).
        """
        # Combine the last state and all specified non-semantic features into a list
        features = [last_state]
        for feature in self.features:
            features.append(inputs[feature])  # Append each feature tensor from the input dictionary

        # Concatenate all features along the last dimension
        x = torch.cat(features, dim=-1)

        # Apply layer normalization, ReLU activation, and dropout
        x = self.dropout(F.relu(self.layernorm(x)))

        # Pass the fused features through the linear transformation
        x = self.linear(x)
        return x


# Additive Attention Mechanism
class Additive_attention(nn.Module):
    """
    A PyTorch implementation of the additive attention mechanism.
    This class calculates attention weights based on learned representations
    and uses them to produce a weighted sum of input features.

    Args:
        input_dim (int): The dimension of the input features (E).
        hidden_dim (int): The dimension of the hidden layer used in attention scoring.
    """
    def __init__(self, input_dim, hidden_dim):
        super(Additive_attention, self).__init__()

        # Initialize input and hidden dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear transformation layer for projecting input features
        self.linear = nn.Linear(self.input_dim, self.hidden_dim)

        # Attention weight parameter, initialized as zeros
        self.weight = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, inputs, mask=None, eps=1e-8):
        """
        Forward pass of the additive attention mechanism.

        Args:
            inputs (torch.Tensor): The input tensor with shape (batch_size, seq_len, input_dim).
            mask (torch.Tensor, optional): A mask tensor with shape (batch_size, seq_len),
                where True indicates positions to be masked (ignored in attention).
                Defaults to None.
            eps (float, optional): A small value to handle numerical stability. Default is 1e-8.

        Returns:
            torch.Tensor: The output tensor after applying the attention mechanism,
            with shape (batch_size, input_dim).
        """
        # Apply a linear transformation followed by a tanh activation to the inputs
        q = torch.tanh(self.linear(inputs))  # Shape: (batch_size, seq_len, hidden_dim)

        # Compute attention scores by applying a learned weight vector
        s_n = torch.matmul(q, self.weight)  # Shape: (batch_size, seq_len)

        # Apply the mask if provided, setting masked positions to a small value (eps)
        if mask is not None:
            s_n = torch.masked_fill(s_n, mask, eps)

        # Compute attention weights using softmax along the sequence length dimension
        alpha = torch.softmax(s_n, dim=-1)  # Shape: (batch_size, seq_len)

        # Apply attention weights to the inputs
        weighted_outputs = inputs * alpha.unsqueeze(-1)  # Shape: (batch_size, seq_len, input_dim)

        # Sum the weighted inputs along the sequence length dimension to get the final output
        outputs = torch.sum(weighted_outputs, dim=1)  # Shape: (batch_size, input_dim)

        return outputs


# TextCNN model for text classification
class TextCNN(nn.Module):
    """
    A PyTorch implementation of the TextCNN model for text classification tasks.
    The model integrates CNN-based encoders for processing different types of text
    inputs (e.g., section titles, subtitles, and content) and fuses non-semantic features
    for enhanced classification performance.

    Args:
        vocab_size (int): The size of the vocabulary (number of unique tokens).
        embed_dim (int): The dimensionality of the word embeddings.
        hidden_dim (int): The number of hidden units in the CNN encoders.
        num_classes (int): The number of output classes for classification.
        kernel_sizes (list, optional): A list of kernel sizes for the CNN layers.
            Defaults to [1, 2, 3, 4].
        padding_idx (int, optional): The index for padding tokens in the embedding layer. Default is 0.
        dropout (float, optional): The dropout rate applied for regularization. Default is 0.1.
        device (torch.device, optional): The device on which the model will run. Default is None.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 kernel_sizes=None, padding_idx=0, dropout=0.1, device=None):
        super(TextCNN, self).__init__()

        # Model parameters
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.dropout_rate = dropout
        self.device = device
        self.kernel_sizes = [1, 2, 3, 4] if kernel_sizes is None else kernel_sizes

        # Embedding layer for mapping input tokens to dense vectors
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embed_dim,
                                      padding_idx=self.padding_idx)

        # CNN encoders for processing section title, subtitle, and text
        self.sec_title_cnn_encoder = CNNEncoder(
            input_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout_rate
        )

        self.sec_subtitle_cnn_encoder = CNNEncoder(
            input_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout_rate
        )

        self.sec_text_cnn_encoder = CNNEncoder(
            input_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout_rate
        )

        # Layer normalization for stabilizing feature concatenation
        self.layernorm = nn.LayerNorm(normalized_shape=(3 * self.hidden_dim))

        # Non-semantic feature fusion module
        self.non_semantic_feature_fusion = Non_semantic_feature_fusion(
            input_dim=3 * self.hidden_dim,
            output_dim=self.hidden_dim
        )

        # Final output layer for classification
        self.output = nn.Linear(in_features=self.hidden_dim, out_features=self.num_classes)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        """
        Forward pass of the TextCNN model.

        Args:
            inputs (dict): A dictionary containing the input data. The keys include:
                - 'sec_title_ids': Token IDs for section titles (batch_size, seq_len).
                - 'sec_subtitle_ids': Token IDs for section subtitles (batch_size, seq_len).
                - 'sec_text_ids': Token IDs for section content (batch_size, seq_len).
                - Additional keys for non-semantic features.

        Returns:
            torch.Tensor: The output logits for classification, with shape (batch_size, num_classes).
        """
        # Encode section titles using the CNN encoder
        sec_name_cnn_result = self.sec_title_cnn_encoder(self.embedding(inputs['sec_title_ids']))

        # Encode section subtitles using the CNN encoder
        sec_header_cnn_result = self.sec_subtitle_cnn_encoder(self.embedding(inputs['sec_subtitle_ids']))

        # Encode section text using the CNN encoder
        sec_text_cnn_result = self.sec_text_cnn_encoder(self.embedding(inputs['sec_text_ids']))

        # Concatenate the encoded features from the three encoders and apply normalization and dropout
        cat_x = self.dropout(F.relu(self.layernorm(
            torch.cat([sec_name_cnn_result, sec_header_cnn_result, sec_text_cnn_result], dim=-1)
        )))

        # Fuse the concatenated features with non-semantic features
        x = self.non_semantic_feature_fusion(cat_x, inputs)

        # Apply the output layer to obtain the final classification logits
        outputs = self.output(x)
        return outputs



# BiLSTM model for text classification
class BiLSTM(nn.Module):
    """
    A PyTorch implementation of a BiLSTM-based model for text classification tasks.
    The model processes section titles and subtitles using CNN encoders and leverages
    a BiLSTM for encoding section content. Non-semantic features are also integrated
    for enhanced performance.

    Args:
        vocab_size (int): The size of the vocabulary (number of unique tokens).
        embed_dim (int): The dimensionality of the word embeddings.
        hidden_dim (int): The number of hidden units in the LSTM layers.
        num_classes (int): The number of output classes for classification.
        num_layers (int, optional): The number of LSTM layers. Default is 1.
        kernel_sizes (list, optional): A list of kernel sizes for the CNN layers.
            Defaults to [1, 2, 3, 4].
        padding_idx (int, optional): The index for padding tokens in the embedding layer. Default is 0.
        dropout (float, optional): The dropout rate applied for regularization. Default is 0.1.
        device (torch.device, optional): The device on which the model will run. Default is None.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, kernel_sizes=None, padding_idx=0, dropout=0.1, device=None):
        super(BiLSTM, self).__init__()

        # Model parameters
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.dropout_rate = dropout
        self.device = device
        self.kernel_sizes = [1, 2, 3, 4] if kernel_sizes is None else kernel_sizes

        # Embedding layer for mapping input tokens to dense vectors
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embed_dim,
                                      padding_idx=self.padding_idx)

        # CNN encoders for processing section title and subtitle
        self.sec_title_cnn_encoder = CNNEncoder(
            input_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout_rate
        )

        self.sec_subtitle_cnn_encoder = CNNEncoder(
            input_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout_rate
        )

        # BiLSTM for encoding section content
        self.bilstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=True
        )

        # Layer normalization for stabilizing feature concatenation
        self.layernorm = nn.LayerNorm(normalized_shape=(4 * self.hidden_dim))

        # Non-semantic feature fusion module
        self.non_semantic_feature_fusion = Non_semantic_feature_fusion(
            input_dim=4 * self.hidden_dim,
            output_dim=self.hidden_dim
        )

        # Final output layer for classification
        self.output = nn.Linear(in_features=self.hidden_dim, out_features=self.num_classes)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        """
        Forward pass of the BiLSTM model.

        Args:
            inputs (dict): A dictionary containing the input data. The keys include:
                - 'sec_title_ids': Token IDs for section titles (batch_size, seq_len).
                - 'sec_subtitle_ids': Token IDs for section subtitles (batch_size, seq_len).
                - 'sec_text_ids': Token IDs for section content (batch_size, seq_len).
                - Additional keys for non-semantic features.

        Returns:
            torch.Tensor: The output logits for classification, with shape (batch_size, num_classes).
        """
        # Encode section titles using the CNN encoder
        sec_name_cnn_result = self.sec_title_cnn_encoder(self.embedding(inputs['sec_title_ids']))

        # Encode section subtitles using the CNN encoder
        sec_header_cnn_result = self.sec_subtitle_cnn_encoder(self.embedding(inputs['sec_subtitle_ids']))

        # Encode section content using the BiLSTM
        sec_text_ids = inputs['sec_text_ids'].transpose(0, 1)  # Transpose for LSTM (seq_len, batch_size)
        embed_sec_text = self.embedding(sec_text_ids)  # Embedding layer output
        _, (hn, _) = self.bilstm(embed_sec_text)  # Get hidden states from BiLSTM

        # Split hidden states into forward and backward components and concatenate
        x1, x2 = torch.chunk(hn, 2, dim=0)
        sec_text_result = torch.cat([x1, x2], dim=-1).squeeze(0)

        # Concatenate features from title, subtitle, and content encodings
        cat_x = self.dropout(F.relu(self.layernorm(torch.cat(
            [sec_name_cnn_result, sec_header_cnn_result, sec_text_result], dim=-1
        ))))

        # Fuse concatenated features with non-semantic features
        x = self.non_semantic_feature_fusion(cat_x, inputs)

        # Apply the output layer to obtain the final classification logits
        outputs = self.output(x)
        return outputs


# BiLSTM model with attention mechanism
class BiLSTM_Attention(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, kernel_sizes=None, padding_idx=0, dropout=0.1, device=None):
        super(BiLSTM_Attention, self).__init__()

        # Model configuration parameters
        self.vocab_size = vocab_size  # Vocabulary size
        self.embed_dim = embed_dim  # Embedding dimension
        self.hidden_dim = hidden_dim  # Hidden layer dimension
        self.num_classes = num_classes  # Number of output classes
        self.num_layers = num_layers  # Number of LSTM layers
        self.padding_idx = padding_idx  # Padding index for embeddings
        self.dropout_rate = dropout  # Dropout rate
        self.device = device  # Device for computation (CPU/GPU)
        self.kernel_sizes = [1, 2, 3, 4] if kernel_sizes is None else kernel_sizes  # CNN kernel sizes

        # Embedding layer to transform input indices to dense vectors
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embed_dim,
                                      padding_idx=self.padding_idx)

        # CNN encoder for processing section titles
        self.sec_title_cnn_encoder = CNNEncoder(input_dim=self.embed_dim,
                                                hidden_dim=self.hidden_dim,
                                                output_dim=self.hidden_dim,
                                                kernel_sizes=self.kernel_sizes,
                                                dropout=self.dropout_rate)

        # CNN encoder for processing section subtitles
        self.sec_subtitle_cnn_encoder = CNNEncoder(input_dim=self.embed_dim,
                                                   hidden_dim=self.hidden_dim,
                                                   output_dim=self.hidden_dim,
                                                   kernel_sizes=self.kernel_sizes,
                                                   dropout=self.dropout_rate)

        # Bidirectional LSTM for processing section content
        self.bilstm = nn.LSTM(self.embed_dim, self.hidden_dim,
                              num_layers=self.num_layers, bidirectional=True)

        # Additive attention mechanism for weighted feature aggregation
        self.attention = Additive_attention(self.hidden_dim * 2, self.hidden_dim)

        # Layer normalization for stabilizing training and improving performance
        self.layernorm = nn.LayerNorm(normalized_shape=(4 * self.hidden_dim))

        # Module for fusing non-semantic features with the aggregated features
        self.non_semantic_feature_fusion = Non_semantic_feature_fusion(
            input_dim=4 * self.hidden_dim,
            output_dim=self.hidden_dim)

        # Fully connected output layer for classification
        self.output = nn.Linear(in_features=self.hidden_dim, out_features=self.num_classes)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        # Process section title through CNN encoder
        sec_name_cnn_result = self.sec_title_cnn_encoder(self.embedding(inputs['sec_title_ids']))

        # Process section subtitle through CNN encoder
        sec_header_cnn_result = self.sec_subtitle_cnn_encoder(self.embedding(inputs['sec_subtitle_ids']))

        # Process section content through embedding and BiLSTM
        sec_text_ids = inputs['sec_text_ids'].transpose(0, 1)  # Transpose for LSTM compatibility
        embed_sec_text = self.embedding(sec_text_ids)  # Embed the section content
        x, _ = self.bilstm(embed_sec_text)  # BiLSTM output

        # Apply attention mechanism to the BiLSTM output
        sec_text_result = self.attention(x.transpose(0, 1), inputs['sec_text_ids'].eq(0))

        # Concatenate title, subtitle, and content features
        cat_x = self.dropout(F.relu(self.layernorm(torch.cat(
            [sec_name_cnn_result, sec_header_cnn_result, sec_text_result], dim=-1))))

        # Fuse non-semantic features with concatenated features
        x = self.non_semantic_feature_fusion(cat_x, inputs)

        # Final output through the fully connected layer
        outputs = self.output(x)

        return outputs


