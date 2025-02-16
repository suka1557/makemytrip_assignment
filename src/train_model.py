import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append("./")

from utils.calculate_metrics import compute_metrics
from utils.save_embedding import save_embeddings_file
from utils.pr_curve import plot_precision_recall_get_optimal_threshold
from models_architecture.model1 import SearchModel1  as NNModel# Change the filename here to import correct model
from configs.main_config import (
    EPOCHS,
    NO_OF_CITY_PAIRS,
    CITY_PAIR_EMBEDDING_DIMENSION,
    COUNT_NUMERICAL_COLUMNS,
    MODEL_NAME,
    CLASS_WEIGHTS,
    PATIENCE,
)
from src.prepare_data import PrepareTrainTest

# 🚀 Training Loop with Validation Loss
def train_model(dataloader, model, criterion, optimizer, class_weights, num_epochs=EPOCHS, patience=PATIENCE):
    for epoch in range(num_epochs):
        model.train()
        batch_count = 0

        #keep list for comparing predictions at EPOCH level
        y_train_true = []
        y_train_pred = []
        y_val_true = []
        y_val_pred = []

        for X_train, y_train, X_val, y_val in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_count += 1

            # Add true labels in lists
            y_train_true.extend(y_train)
            y_val_true.extend(y_val)

            # Extract city pair index & numerical features for Training
            X_city_train = X_train[:, 0].long().to(device)  
            X_numerical_train = X_train[:, 1:].float().to(device)  
            y_train = y_train.float().to(device).unsqueeze(1)  

            # Compute per-sample class weights
            train_sample_weights = y_train * class_weights[1] + (1 - y_train) * class_weights[0]

            # 🔄 Forward Pass - Training
            optimizer.zero_grad()
            train_outputs = model(X_numerical_train, X_city_train)
            train_loss = criterion(train_outputs, y_train)
            train_loss = (train_loss * train_sample_weights).mean()  # Apply weights and average
            train_loss.backward()
            optimizer.step()

            # Add predicted train prob
            y_train_pred.extend(train_outputs)

            # 🏁 Validation Step
            model.eval()  
            with torch.no_grad():
                X_city_val = X_val[:, 0].long().to(device)  
                X_numerical_val = X_val[:, 1:].float().to(device)  
                y_val = y_val.float().to(device).unsqueeze(1)  

                val_outputs = model(X_numerical_val, X_city_val)

                #Add predicted val to list
                y_val_pred.extend(val_outputs)

                # Compute per-sample class weights for validation
                val_sample_weights = y_val * class_weights[1] + (1 - y_val) * class_weights[0]

                # Compute weighted validation loss
                val_loss = criterion(val_outputs, y_val)
                val_loss = (val_loss * val_sample_weights).mean()  # Apply weights and average


            model.train()  # Switch back to training mode

            # 🔹 Print metrics for each batch
            print(f"Epoch {epoch+1} | Batch {batch_count} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")



        # Get EPOCH level metrics
        y_train_true = [t.item() for t in y_train_true]
        y_train_pred = [t.item() for t in y_train_pred]
        y_val_true = [t.item() for t in y_val_true]
        y_val_pred = [t.item() for t in y_val_pred]

        epoch_optimal_threshold = plot_precision_recall_get_optimal_threshold(y_train_true, y_train_pred, class_weights, epoch)
        training_metrics = compute_metrics(y_train_true, y_train_pred,  epoch_optimal_threshold)
        validation_metrics = compute_metrics(y_val_true, y_val_pred, epoch_optimal_threshold)
        #Save PR curve for validation set
        _ = plot_precision_recall_get_optimal_threshold( y_val_true, y_val_pred, class_weights, epoch, save_plot=True)

        print(f"Epoch {epoch+1} Summary -> "
        f"Train Acc: {training_metrics['accuracy']:.4f}, "
        f"Train Precision: {training_metrics['precision']:.4f}, Train Recall: {training_metrics['recall']:.4f}, Train ROC-AUC: {training_metrics['roc_auc']:.4f} | "
        f"Val Acc: {validation_metrics['accuracy']:.4f}, "
        f"Val Precision: {validation_metrics['precision']:.4f}, Val Recall: {validation_metrics['recall']:.4f}, Val ROC-AUC: {validation_metrics['roc_auc']:.4f}"
        )
        
        #Save embedding after completion of 1 epoch
        save_embeddings_file(model=model)

        #Save model after completion of 1 epoch
        torch.save(model.state_dict(), MODEL_NAME)

        

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NNModel(num_numerical=COUNT_NUMERICAL_COLUMNS, num_pairs=NO_OF_CITY_PAIRS, emb_dim=CITY_PAIR_EMBEDDING_DIMENSION).to(device)
    
    class_weights = torch.tensor([CLASS_WEIGHTS[0], CLASS_WEIGHTS[1]], dtype=torch.float, device=device)

    criterion = nn.BCEWithLogitsLoss(reduction='none')  # Compute per sample loss 
    
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 🔄 SGD with Momentum
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  

    dataset = PrepareTrainTest()
    dataloader = DataLoader(dataset, batch_size=None)  # No need to batch, already handled

    #train the model
    train_model(dataloader, model, criterion, optimizer, class_weights, EPOCHS)
