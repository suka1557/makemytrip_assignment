import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os, sys

sys.path.append("./")

from configs.main_config import (
    COUNT_NUMERICAL_COLUMNS,
    NO_OF_CITY_PAIRS,
    CITY_PAIR_EMBEDDING_DIMENSION
)

class SearchModel2(nn.Module):
    def __init__(self, num_numerical=COUNT_NUMERICAL_COLUMNS, num_pairs=NO_OF_CITY_PAIRS, emb_dim=CITY_PAIR_EMBEDDING_DIMENSION):
        super(SearchModel2, self).__init__()

        self.embedding = nn.Embedding(num_pairs, emb_dim)  # City pair embeddings

        # Fully Connected Layers (Increased Depth)
        self.fc1 = nn.Linear(num_numerical + emb_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)  # Output Layer

        # Activation, Dropout & BatchNorm
        self.leaky_relu = nn.LeakyReLU(0.1)  # Better than ReLU for deep networks
        self.dropout = nn.Dropout(0.4)  # Increased to prevent overfitting
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.batchnorm4 = nn.BatchNorm1d(256)
        self.batchnorm5 = nn.BatchNorm1d(128)
        self.batchnorm6 = nn.BatchNorm1d(64)
        self.sigmoid = nn.Sigmoid()  # Binary classification output

    def forward(self, x_num, x_pair):
        emb = self.embedding(x_pair)  # Get city pair embedding
        x = torch.cat([x_num, emb], dim=1)  # Concatenate numerical + embeddings

        # Fully Connected Forward Propagation with BatchNorm, Dropout, and LeakyReLU
        x = self.leaky_relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.batchnorm3(self.fc3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.batchnorm4(self.fc4(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.batchnorm5(self.fc5(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.batchnorm6(self.fc6(x)))
        x = self.dropout(x)

        x = self.sigmoid(self.fc_out(x))  # Final output
        return x
