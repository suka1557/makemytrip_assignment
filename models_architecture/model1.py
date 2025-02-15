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

# 🏗 Define PyTorch Model
class SearchModel1(nn.Module):
    def __init__(self, num_numerical=COUNT_NUMERICAL_COLUMNS, num_pairs=NO_OF_CITY_PAIRS, emb_dim=CITY_PAIR_EMBEDDING_DIMENSION):
        super(SearchModel1, self).__init__()
        self.embedding = nn.Embedding(num_pairs, emb_dim)  # Embedding for city pairs
        self.fc1 = nn.Linear(num_numerical + emb_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_num, x_pair):
        emb = self.embedding(x_pair)  # Get embedding for city pair
        x = torch.cat([x_num, emb], dim=1)  # Concatenate numerical features with embedding
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x
