import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append("./")

from utils.calculate_metrics import compute_metrics
from utils.save_embedding import save_embeddings_file
from models_architecture.model2 import SearchModel2  as NNModel# Change the filename here to import correct model
from configs.main_config import (
    EPOCHS,
    NO_OF_CITY_PAIRS,
    CITY_PAIR_EMBEDDING_DIMENSION,
    COUNT_NUMERICAL_COLUMNS,
    MODEL_NAME,
)
from src.prepare_data import PrepareTrainTest

# 🚀 Training Loop with Validation Loss
def train_model(dataloader, model, criterion, optimizer, num_epochs=EPOCHS):
    for epoch in range(num_epochs):
        model.train()
        total_train_metrics = {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "roc_auc": 0}
        total_val_metrics = {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "roc_auc": 0}
        batch_count = 0

        for X_train, y_train, X_val, y_val in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_count += 1

            # Extract city pair index & numerical features for Training
            X_city_train = X_train[:, 0].long().to(device)  
            X_numerical_train = X_train[:, 1:].float().to(device)  
            y_train = y_train.float().to(device).unsqueeze(1)  

            # 🔄 Forward Pass - Training
            optimizer.zero_grad()
            train_outputs = model(X_numerical_train, X_city_train)
            train_loss = criterion(train_outputs, y_train)
            train_loss.backward()
            optimizer.step()

            # Compute Training Metrics
            train_metrics = compute_metrics(y_train, train_outputs, criterion)
            for key in total_train_metrics:
                total_train_metrics[key] += train_metrics[key]

            # 🏁 Validation Step
            model.eval()  
            with torch.no_grad():
                X_city_val = X_val[:, 0].long().to(device)  
                X_numerical_val = X_val[:, 1:].float().to(device)  
                y_val = y_val.float().to(device).unsqueeze(1)  

                val_outputs = model(X_numerical_val, X_city_val)

                # Compute Validation Metrics
                val_metrics = compute_metrics(y_val, val_outputs, criterion)
                for key in total_val_metrics:
                    total_val_metrics[key] += val_metrics[key]

            model.train()  # Switch back to training mode

            # 🔹 Print metrics for each batch
            print(f"Epoch {epoch+1} | Batch {batch_count} -> Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")

        # 🎯 Compute Epoch-Wise Averages
        avg_train_metrics = {key: value / batch_count for key, value in total_train_metrics.items()}
        avg_val_metrics = {key: value / batch_count for key, value in total_val_metrics.items()}

        print(f"Epoch {epoch+1} Summary -> "
              f"Train Loss: {avg_train_metrics['loss']:.4f}, Train Acc: {avg_train_metrics['accuracy']:.4f}, "
              f"Train Precision: {avg_train_metrics['precision']:.4f}, Train Recall: {avg_train_metrics['recall']:.4f}, Train ROC-AUC: {avg_train_metrics['roc_auc']:.4f} | "
              f"Val Loss: {avg_val_metrics['loss']:.4f}, Val Acc: {avg_val_metrics['accuracy']:.4f}, "
              f"Val Precision: {avg_val_metrics['precision']:.4f}, Val Recall: {avg_val_metrics['recall']:.4f}, Val ROC-AUC: {avg_val_metrics['roc_auc']:.4f}"
             )
        
        #Save embedding after completion of 1 epoch
        save_embeddings_file(model=model)

        #Save model after completion of 1 epoch
        torch.save(model.state_dict(), MODEL_NAME)

        

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NNModel(num_numerical=COUNT_NUMERICAL_COLUMNS, num_pairs=NO_OF_CITY_PAIRS, emb_dim=CITY_PAIR_EMBEDDING_DIMENSION).to(device)

    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = PrepareTrainTest()
    dataloader = DataLoader(dataset, batch_size=None)  # No need to batch, already handled

    #train the model
    train_model(dataloader, model, criterion, optimizer, EPOCHS)
